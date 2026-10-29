use futures::{Stream, StreamExt};
use goose_provider_types::{
    conversation::{message::Message, token_usage::ProviderUsage},
    errors::ProviderError,
    formats::{anthropic, google, openai, openai_responses},
};
use serde_json::{json, Value};
use std::pin::Pin;

type Chunk = (Option<Message>, Option<ProviderUsage>);
type Parsed = Pin<Box<dyn Stream<Item = anyhow::Result<Chunk>> + Send>>;

#[derive(Clone, Copy, Debug)]
enum Format {
    Chat,
    Responses,
    Anthropic,
    Google,
}

const FORMATS: [Format; 4] = [
    Format::Chat,
    Format::Responses,
    Format::Anthropic,
    Format::Google,
];

fn parse(format: Format, lines: Vec<anyhow::Result<String>>) -> Parsed {
    let lines = futures::stream::iter(lines);
    match format {
        Format::Chat => Box::pin(openai::response_to_streaming_message(lines)),
        Format::Responses => Box::pin(openai_responses::responses_api_to_streaming_message(lines)),
        Format::Anthropic => Box::pin(anthropic::response_to_streaming_message(lines)),
        Format::Google => Box::pin(google::response_to_streaming_message(lines)),
    }
}

fn frame(value: Value) -> String {
    format!("data: {value}")
}

fn text(format: Format) -> Vec<String> {
    match format {
        Format::Chat => vec![
            frame(
                json!({"id":"reply","model":"model","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}),
            ),
            frame(json!({"id":"reply","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]})),
        ],
        Format::Responses => vec![
            frame(
                json!({"type":"response.output_text.delta","sequence_number":1,"item_id":"item","output_index":0,"content_index":0,"delta":"Hello"}),
            ),
            frame(
                json!({"type":"response.completed","sequence_number":2,"response":{"id":"reply","object":"response","created_at":1,"status":"completed","model":"model","output":[]}}),
            ),
        ],
        Format::Anthropic => vec![
            frame(
                json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}),
            ),
            frame(json!({"type":"message_stop"})),
        ],
        Format::Google => vec![
            frame(json!({"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"}]}}]})),
            frame(json!({"candidates":[{"finishReason":"STOP"}]})),
        ],
    }
}

fn tool(format: Format) -> Vec<String> {
    match format {
        Format::Chat => vec![frame(
            json!({"id":"reply","model":"model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call","type":"function","function":{"name":"lookup","arguments":"{\"path\":\"file\"}"}}]},"finish_reason":"tool_calls"}]}),
        )],
        Format::Responses => vec![frame(
            json!({"type":"response.completed","sequence_number":1,"response":{"id":"reply","object":"response","created_at":1,"status":"completed","model":"model","output":[{"type":"function_call","id":"item","status":"completed","call_id":"call","name":"lookup","arguments":"{\"path\":\"file\"}"}]}}),
        )],
        Format::Anthropic => vec![
            frame(
                json!({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call","name":"lookup","input":{}}}),
            ),
            frame(
                json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"file\"}"}}),
            ),
            frame(json!({"type":"content_block_stop","index":0})),
            frame(json!({"type":"message_stop"})),
        ],
        Format::Google => vec![frame(
            json!({"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"id":"call","name":"lookup","args":{"path":"file"}}}]},"finishReason":"STOP"}]}),
        )],
    }
}

#[tokio::test]
async fn accepts_terminal_without_usage() {
    for format in FORMATS {
        let mut stream = parse(format, text(format).into_iter().map(Ok).collect());
        let mut output = String::new();
        while let Some(chunk) = stream.next().await {
            let (message, usage) = chunk.unwrap_or_else(|error| panic!("{format:?}: {error}"));
            if let Some(message) = message {
                output.push_str(&message.as_concat_text());
            }
            if let Some(usage) = usage {
                assert_eq!(usage.usage.input_tokens, None, "{format:?}");
                assert_eq!(usage.usage.output_tokens, None, "{format:?}");
            }
        }
        assert_eq!(output, "Hello", "{format:?}");
    }
}

#[tokio::test]
async fn retains_completed_tool_calls() {
    for format in FORMATS {
        let mut stream = parse(format, tool(format).into_iter().map(Ok).collect());
        let mut calls = Vec::new();
        while let Some(chunk) = stream.next().await {
            let (message, _) = chunk.unwrap_or_else(|error| panic!("{format:?}: {error}"));
            if let Some(message) = message {
                calls.extend(message.content.into_iter().filter_map(|content| {
                    content.as_tool_request().map(|request| request.clone())
                }));
            }
        }
        assert_eq!(calls.len(), 1, "{format:?}");
        let call = calls[0].tool_call.as_ref().unwrap();
        assert_eq!(call.name, "lookup", "{format:?}");
        assert_eq!(
            call.arguments.as_ref().unwrap()["path"],
            "file",
            "{format:?}"
        );
    }
}

#[tokio::test]
async fn rejects_eof_after_text() {
    for format in FORMATS {
        let mut lines = text(format);
        lines.pop();
        let mut stream = parse(format, lines.into_iter().map(Ok).collect());
        let mut output = String::new();
        let mut errors = Vec::new();
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok((Some(message), _)) => output.push_str(&message.as_concat_text()),
                Ok(_) => {}
                Err(error) => errors.push(ProviderError::from_stream_error(error)),
            }
        }
        assert_eq!(output, "Hello", "{format:?}");
        assert!(
            matches!(errors.as_slice(), [ProviderError::NetworkError(_)]),
            "{format:?}: {errors:?}"
        );
    }
}

#[tokio::test]
async fn rejects_eof_before_output() {
    for format in FORMATS {
        let mut stream = parse(format, Vec::new());
        let error = stream.next().await.unwrap().unwrap_err();
        assert!(
            matches!(
                ProviderError::from_stream_error(error),
                ProviderError::NetworkError(_)
            ),
            "{format:?}"
        );
        assert!(stream.next().await.is_none());
    }
}

#[tokio::test]
async fn preserves_transport_errors() {
    for format in FORMATS {
        let mut lines: Vec<_> = text(format).into_iter().take(1).map(Ok).collect();
        let expected = ProviderError::NetworkError("connection closed".into());
        lines.push(Err(anyhow::Error::new(expected.clone())));
        let mut stream = parse(format, lines);
        let mut failure = None;
        while let Some(chunk) = stream.next().await {
            if let Err(error) = chunk {
                failure = Some(ProviderError::from_stream_error(error));
            }
        }
        assert_eq!(failure, Some(expected), "{format:?}");
    }
}
