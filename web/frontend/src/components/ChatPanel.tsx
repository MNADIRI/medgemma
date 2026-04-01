import React, { useCallback, useEffect, useRef, useState } from "react";
import type { ChatMessage } from "../types";

interface Props {
  messages: ChatMessage[];
  onSend: (message: string) => void;
  isLoading: boolean;
  disabled: boolean;
  selectedCount: number;
}

export default function ChatPanel({ messages, onSend, isLoading, disabled, selectedCount }: Props) {
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      const text = input.trim();
      if (!text || isLoading) return;
      onSend(text);
      setInput("");
    },
    [input, isLoading, onSend]
  );

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        background: "#111",
        borderRadius: 8,
        border: "1px solid #333",
        overflow: "hidden",
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: "12px 16px",
          borderBottom: "1px solid #333",
          fontSize: 14,
          fontWeight: 600,
          color: "#ddd",
        }}
      >
        MedGemma Chat
      </div>

      {/* Messages */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          padding: 16,
          display: "flex",
          flexDirection: "column",
          gap: 12,
        }}
      >
        {messages.length === 0 && !isLoading && (
          <div style={{ color: "#555", textAlign: "center", marginTop: 40, fontSize: 14 }}>
            {disabled
              ? "Upload DICOM files to start chatting"
              : "Select slices and ask MedGemma a question"}
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            style={{
              alignSelf: msg.role === "user" ? "flex-end" : "flex-start",
              maxWidth: "85%",
              padding: "10px 14px",
              borderRadius: 12,
              background: msg.role === "user" ? "#1a3a5c" : "#222",
              color: msg.role === "user" ? "#ddeeff" : "#ccc",
              fontSize: 14,
              lineHeight: 1.5,
              whiteSpace: "pre-wrap",
            }}
          >
            {msg.role === "assistant" && (
              <div style={{ fontSize: 11, color: "#4a90d9", marginBottom: 4, fontWeight: 600 }}>
                MedGemma
              </div>
            )}
            {msg.content}
          </div>
        ))}

        {isLoading && (
          <div
            style={{
              alignSelf: "flex-start",
              padding: "10px 14px",
              borderRadius: 12,
              background: "#222",
              color: "#888",
              fontSize: 14,
            }}
          >
            <span className="dots">Analyzing</span>
            <style>{`
              .dots::after {
                content: '';
                animation: dots 1.5s steps(4, end) infinite;
              }
              @keyframes dots {
                0% { content: ''; }
                25% { content: '.'; }
                50% { content: '..'; }
                75% { content: '...'; }
              }
            `}</style>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Slice count warning */}
      {selectedCount > 20 && (
        <div style={{ padding: "6px 12px", background: "#2a2000", color: "#f0c040", fontSize: 12, borderTop: "1px solid #333" }}>
          {selectedCount} slices selected — the backend will sample 20 to fit in memory.
        </div>
      )}

      {/* Input */}
      <form
        onSubmit={handleSubmit}
        style={{
          display: "flex",
          gap: 8,
          padding: 12,
          borderTop: "1px solid #333",
        }}
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={
            disabled
              ? "Upload DICOM files first..."
              : selectedCount === 0
              ? "Select slices first..."
              : `Ask about ${selectedCount} selected slice${selectedCount > 1 ? "s" : ""}...`
          }
          disabled={disabled || isLoading || selectedCount === 0}
          style={{
            flex: 1,
            padding: "10px 14px",
            borderRadius: 8,
            border: "1px solid #444",
            background: "#1a1a1a",
            color: "#eee",
            fontSize: 14,
            outline: "none",
          }}
        />
        <button
          type="submit"
          disabled={disabled || isLoading || selectedCount === 0 || !input.trim()}
          style={{
            padding: "10px 20px",
            borderRadius: 8,
            border: "none",
            background: disabled || isLoading || selectedCount === 0 ? "#333" : "#4a90d9",
            color: "#fff",
            cursor: disabled || isLoading || selectedCount === 0 ? "not-allowed" : "pointer",
            fontSize: 14,
            fontWeight: 600,
          }}
        >
          Send
        </button>
      </form>
    </div>
  );
}
