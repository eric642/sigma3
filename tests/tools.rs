use sigma::types::chat::{
    ChatCompletionMessageCustomToolCall, ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCalls, ChatCompletionNamedToolChoice, ChatCompletionTool,
    ChatCompletionToolChoiceOption, ChatCompletionTools, CustomTool, CustomToolChatCompletions,
    CustomToolProperties, CustomToolPropertiesFormat, ToolChoiceOptions,
};
use sigma::types::shared::{
    CustomGrammarFormatParam, FunctionCall, FunctionName, FunctionObject, GrammarSyntax,
};

#[test]
fn function_tool_serializes_with_type_function() {
    let v = ChatCompletionTools::Function(ChatCompletionTool {
        function: FunctionObject {
            name: "f".into(),
            description: None,
            parameters: None,
            strict: None,
        },
    });
    let s = serde_json::to_string(&v).unwrap();
    assert_eq!(s, r#"{"type":"function","function":{"name":"f"}}"#);
}

#[test]
fn custom_tool_default_format_text() {
    let v = ChatCompletionTools::Custom(CustomToolChatCompletions {
        custom: CustomToolProperties {
            name: "c".into(),
            description: None,
            format: CustomToolPropertiesFormat::Text,
        },
    });
    let s = serde_json::to_string(&v).unwrap();
    assert!(s.contains(r#""type":"custom""#));
    assert!(s.contains(r#""format":{"type":"text"}"#));
}

#[test]
fn custom_tool_grammar_format() {
    let f = CustomToolPropertiesFormat::Grammar {
        grammar: CustomGrammarFormatParam {
            definition: "[a-z]+".into(),
            syntax: GrammarSyntax::Regex,
        },
    };
    let s = serde_json::to_string(&f).unwrap();
    assert_eq!(
        s,
        r#"{"type":"grammar","grammar":{"definition":"[a-z]+","syntax":"regex"}}"#
    );
}

#[test]
fn message_tool_calls_function() {
    let v = ChatCompletionMessageToolCalls::Function(ChatCompletionMessageToolCall {
        id: "call_1".into(),
        function: FunctionCall {
            name: "f".into(),
            arguments: "{}".into(),
        },
    });
    let s = serde_json::to_string(&v).unwrap();
    assert_eq!(
        s,
        r#"{"type":"function","id":"call_1","function":{"name":"f","arguments":"{}"}}"#
    );
}

#[test]
fn message_tool_calls_custom() {
    let v = ChatCompletionMessageToolCalls::Custom(ChatCompletionMessageCustomToolCall {
        id: "call_2".into(),
        custom_tool: CustomTool {
            name: "c".into(),
            input: "x".into(),
        },
    });
    let s = serde_json::to_string(&v).unwrap();
    let back: ChatCompletionMessageToolCalls = serde_json::from_str(&s).unwrap();
    assert_eq!(v, back);
}

#[test]
fn tool_choice_mode_untagged() {
    let v = ChatCompletionToolChoiceOption::Mode(ToolChoiceOptions::Auto);
    let s = serde_json::to_string(&v).unwrap();
    assert_eq!(s, r#""auto""#);
}

#[test]
fn tool_choice_named_function() {
    let v = ChatCompletionToolChoiceOption::Function(ChatCompletionNamedToolChoice {
        function: FunctionName {
            name: "get_weather".into(),
        },
    });
    let s = serde_json::to_string(&v).unwrap();
    assert_eq!(
        s,
        r#"{"type":"function","function":{"name":"get_weather"}}"#
    );
}

#[test]
fn tool_choice_default_is_none() {
    assert_eq!(ToolChoiceOptions::default(), ToolChoiceOptions::None);
}
