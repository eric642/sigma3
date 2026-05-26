use sigma::types::chat::{
    CustomToolCall, CustomToolCallInput, CustomToolDefinition, CustomToolFormat,
    CustomToolProperties, FunctionTool, FunctionToolCall, NamedFunctionToolChoice, ToolCall,
    ToolChoice, ToolChoiceMode, ToolDefinition,
};
use sigma::types::shared::{
    CustomGrammarFormatParam, FunctionCall, FunctionName, FunctionObject, GrammarSyntax,
};

#[test]
fn function_tool_serializes_with_type_function() {
    let v = ToolDefinition::Function(FunctionTool {
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
    let v = ToolDefinition::Custom(CustomToolDefinition {
        custom: CustomToolProperties {
            name: "c".into(),
            description: None,
            format: CustomToolFormat::Text,
        },
    });
    let s = serde_json::to_string(&v).unwrap();
    assert!(s.contains(r#""type":"custom""#));
    assert!(s.contains(r#""format":{"type":"text"}"#));
}

#[test]
fn custom_tool_grammar_format() {
    let f = CustomToolFormat::Grammar {
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
    let v = ToolCall::Function(FunctionToolCall {
        id: "call_1".into(),
        function: FunctionCall {
            name: "f".into(),
            arguments: "{}".into(),
        },
        reasoning: None,
    });
    let s = serde_json::to_string(&v).unwrap();
    assert_eq!(
        s,
        r#"{"type":"function","id":"call_1","function":{"name":"f","arguments":"{}"}}"#
    );
}

#[test]
fn message_tool_calls_custom() {
    let v = ToolCall::Custom(CustomToolCall {
        id: "call_2".into(),
        custom_tool: CustomToolCallInput {
            name: "c".into(),
            input: "x".into(),
        },
        reasoning: None,
    });
    let s = serde_json::to_string(&v).unwrap();
    let back: ToolCall = serde_json::from_str(&s).unwrap();
    assert_eq!(v, back);
}

#[test]
fn tool_choice_mode_untagged() {
    let v = ToolChoice::Mode(ToolChoiceMode::Auto);
    let s = serde_json::to_string(&v).unwrap();
    assert_eq!(s, r#""auto""#);
}

#[test]
fn tool_choice_named_function() {
    let v = ToolChoice::Function(NamedFunctionToolChoice {
        function: FunctionName {
            name: "get_weather".into(),
        },
    });
    let value = serde_json::to_value(&v).unwrap();
    assert_eq!(
        value,
        serde_json::json!({"type":"function","function":{"name":"get_weather"}})
    );
}

#[test]
fn tool_choice_default_is_none() {
    assert_eq!(ToolChoiceMode::default(), ToolChoiceMode::None);
}
