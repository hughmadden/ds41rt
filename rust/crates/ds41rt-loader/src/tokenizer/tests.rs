use super::*;

fn byte_tokenizer() -> tempfile::TempDir {
    // ByteLevel's reversible byte alphabet, with IDs equal to raw bytes.
    let mut extra = 256u32;
    let mut vocab = serde_json::Map::new();
    for byte in 0..=255u32 {
        let character = if (33..=126).contains(&byte) || (161..=172).contains(&byte) || byte >= 174
        {
            char::from_u32(byte).unwrap()
        } else {
            let character = char::from_u32(extra).unwrap();
            extra += 1;
            character
        };
        vocab.insert(character.to_string(), serde_json::json!(byte));
    }
    let dir = tempfile::tempdir().unwrap();
    let value = serde_json::json!({"version":"1.0","truncation":null,"padding":null,
        "added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,
        "decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
        "model":{"type":"WordLevel","vocab":vocab,"unk_token":"?"}});
    std::fs::write(dir.path().join("tokenizer.json"), value.to_string()).unwrap();
    dir
}

#[test]
fn terminal_utf8_matches_lossy_whole_decode_at_every_byte_boundary() {
    let dir = byte_tokenizer();
    for text in [
        "",
        "ASCII",
        "台北 café e\u{301}",
        "👩🏽\u{200d}💻 🦜 🇹🇼",
        "עברית العربية हिन्दी ไทย",
        "\0\r\n\t",
        "�",
        "x��",
        "�x�",
    ] {
        for end in 0..=text.len() {
            let bytes = &text.as_bytes()[..end];
            let mut decoder = streaming_token_decoder(dir.path(), false).unwrap();
            let mut result = String::new();
            for &byte in bytes {
                if let Some(piece) = decoder.step(u32::from(byte)).unwrap() {
                    result.push_str(&piece);
                }
            }
            if let Some(piece) = decoder.finish().unwrap() {
                result.push_str(&piece);
            }
            assert_eq!(
                result,
                String::from_utf8_lossy(bytes),
                "{text:?}, cut {end}"
            );
            assert_eq!(
                decoder.finish().unwrap(),
                None,
                "finish must not emit twice"
            );
        }
    }
}

#[test]
fn terminal_literal_replacement_and_partial_scalar_are_not_dropped() {
    let dir = byte_tokenizer();
    for bytes in [&b"\xef\xbf\xbd"[..], &b"\xf0\x9f\xa6"[..]] {
        let mut decoder = streaming_token_decoder(dir.path(), false).unwrap();
        for &byte in bytes {
            assert!(decoder.step(u32::from(byte)).unwrap().is_none());
        }
        assert_eq!(decoder.finish().unwrap().as_deref(), Some("�"));
        assert_eq!(decoder.step(u32::from(b'A')).unwrap().as_deref(), Some("A"));
        assert_eq!(decoder.finish().unwrap(), None);
    }
}

#[test]
#[ignore = "requires the pinned official tokenizer via DS41RT_UNICODE_MODEL"]
fn official_unicode_roundtrip_and_every_token_cut() {
    let snapshot = PathBuf::from(std::env::var_os("DS41RT_UNICODE_MODEL").expect("model path"));
    let mut cases = vec![
        "台北，世界！",
        "café e\u{301} Å A\u{30a}",
        "👩🏽\u{200d}💻 🦜 🇹🇼",
        "עברית العربية हिन्दी ไทย",
        "�",
        "x��",
        "\0\r\n\t",
        "\u{202e}abc\u{202c}",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<Vec<_>>();
    // Cover every valid scalar's UTF-8 lead/continuation class with a fixed
    // deterministic sample across the Unicode range, plus boundary scalars.
    cases.extend(
        (0..=0x10ffff)
            .step_by(7919)
            .filter_map(char::from_u32)
            .map(|c| format!("A{c}Z")),
    );
    cases.extend(
        [
            0x7f, 0x80, 0x7ff, 0x800, 0xd7ff, 0xe000, 0xffff, 0x10000, 0x10ffff,
        ]
        .into_iter()
        .map(|c| char::from_u32(c).unwrap().to_string()),
    );
    let mut cuts = 0;
    for text in &cases {
        let ids = encode_tokenizer_text(&snapshot, text, false)
            .unwrap()
            .token_ids;
        assert_eq!(
            decode_tokenizer_ids(&snapshot, &ids, false).unwrap().text,
            *text
        );
        for end in 0..=ids.len() {
            let expected = decode_tokenizer_ids(&snapshot, &ids[..end], false)
                .unwrap()
                .text;
            let mut decoder = streaming_token_decoder(&snapshot, false).unwrap();
            let mut result = String::new();
            for &id in &ids[..end] {
                if let Some(piece) = decoder.step(id).unwrap() {
                    result.push_str(&piece);
                }
            }
            if let Some(piece) = decoder.finish().unwrap() {
                result.push_str(&piece);
            }
            assert_eq!(result, expected, "{text:?}, cut {end}");
            cuts += 1;
        }
    }
    println!(
        "official Unicode: {} strings, {cuts} token cuts",
        cases.len()
    );
}
