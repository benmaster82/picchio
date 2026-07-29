#!/usr/bin/env python3
"""Verifica che Harmony renderizzi il turno successivo conservando il prefisso già consumato."""
from openai_harmony import (
    Conversation, HarmonyEncodingName, Message, ReasoningEffort,
    RenderConversationConfig, Role, SystemContent, load_harmony_encoding,
)

ASSISTANT_REPLY = ("<|channel|>analysis<|message|>Breve.<|end|>"
                   "<|start|>assistant<|channel|>final<|message|>Ciao!<|return|>")


def main():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    system = (SystemContent.new()
              .with_reasoning_effort(ReasoningEffort.LOW)
              .with_conversation_start_date("2026-07-29"))
    base = [Message.from_role_and_content(Role.SYSTEM, system),
            Message.from_role_and_content(Role.USER, "Ciao")]

    prompt = encoding.render_conversation_for_completion(
        Conversation.from_messages(base), Role.ASSISTANT)
    output = encoding.encode(ASSISTANT_REPLY, allowed_special="all")
    parsed = encoding.parse_messages_from_completion_tokens(output, Role.ASSISTANT)

    second = base + parsed + [Message.from_role_and_content(Role.USER, "Come stai?")]
    committed = prompt + output
    for message in parsed:
        print("parsed:", message.to_dict())

    for keep_analysis in (False, True):
        config = RenderConversationConfig(auto_drop_analysis=not keep_analysis)
        full = encoding.render_conversation_for_completion(
            Conversation.from_messages(second), Role.ASSISTANT, config)
        shared = 0
        for a, b in zip(full, committed):
            if a != b:
                break
            shared += 1
        prefix_ok = shared == len(committed)
        print(f"\nkeep_analysis={keep_analysis} full2={len(full)} "
              f"shared_prefix={shared}/{len(committed)} prefix_preserving={prefix_ok}")
        print(f"  reuse={shared / len(full):.1%} da_riprocessare={len(full) - shared}")
        print(f"  committed[{shared}:]={encoding.decode_utf8(committed[shared:])!r}")
        print(f"  full[{shared}:]={encoding.decode_utf8(full[shared:])!r}")


if __name__ == "__main__":
    main()
