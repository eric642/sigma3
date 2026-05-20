use sigma::types::chat::{
    GetChatCompletionMessagesOrder, GetChatCompletionMessagesQueryArgs, ListChatCompletionsOrder,
    ListChatCompletionsQueryArgs,
};

#[test]
fn list_query_minimal_skips_none() {
    let q = ListChatCompletionsQueryArgs::default().build().unwrap();
    let s = serde_json::to_string(&q).unwrap();
    assert_eq!(s, "{}");
}

#[test]
fn list_query_with_order() {
    let q = ListChatCompletionsQueryArgs::default()
        .order(ListChatCompletionsOrder::Desc)
        .limit(10u32)
        .build()
        .unwrap();
    let s = serde_json::to_string(&q).unwrap();
    assert_eq!(s, r#"{"limit":10,"order":"desc"}"#);
}

#[test]
fn get_messages_query_round_trip() {
    let q = GetChatCompletionMessagesQueryArgs::default()
        .after("msg_1")
        .order(GetChatCompletionMessagesOrder::Asc)
        .build()
        .unwrap();
    let s = serde_json::to_string(&q).unwrap();
    assert_eq!(s, r#"{"after":"msg_1","order":"asc"}"#);
}
