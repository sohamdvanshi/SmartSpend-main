import { useState, useRef, useEffect } from "react";
import axios from "axios";

const SUGGESTED = [
  "Am I spending too much?",
  "What's my biggest expense category?",
  "How can I save money?",
  "Show me a spending summary",
];

export default function ChatBot() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([
    { role: "bot", text: "Hi! I'm SmartSpend AI 👋 Ask me anything about your expenses!" }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async (text) => {
    const msg = text || input.trim();
    if (!msg) return;
    setInput("");
    setMessages(prev => [...prev, { role: "user", text: msg }]);
    setLoading(true);
    try {
      const res = await axios.post("http://localhost:5000/api/chat", { message: msg });
      setMessages(prev => [...prev, { role: "bot", text: res.data.reply }]);
    } catch {
      setMessages(prev => [...prev, { role: "bot", text: "Sorry, something went wrong. Try again!" }]);
    }
    setLoading(false);
  };

  return (
    <>
      {/* Floating button */}
      <button
        onClick={() => setOpen(!open)}
        style={{
          position: "fixed", bottom: 24, right: 24, zIndex: 1000,
          width: 56, height: 56, borderRadius: "50%",
          background: "#4F46E5", color: "white", fontSize: 24,
          border: "none", cursor: "pointer", boxShadow: "0 4px 12px rgba(0,0,0,0.2)"
        }}
      >
        {open ? "✕" : "💬"}
      </button>

      {/* Chat window */}
      {open && (
        <div style={{
          position: "fixed", bottom: 90, right: 24, zIndex: 1000,
          width: 360, height: 500, background: "white",
          borderRadius: 16, boxShadow: "0 8px 32px rgba(0,0,0,0.15)",
          display: "flex", flexDirection: "column", overflow: "hidden"
        }}>
          {/* Header */}
          <div style={{ background: "#4F46E5", color: "white", padding: "12px 16px", fontWeight: 600 }}>
            💰 SmartSpend AI
          </div>

          {/* Messages */}
          <div style={{ flex: 1, overflowY: "auto", padding: 12, display: "flex", flexDirection: "column", gap: 8 }}>
            {messages.map((m, i) => (
              <div key={i} style={{
                alignSelf: m.role === "user" ? "flex-end" : "flex-start",
                background: m.role === "user" ? "#4F46E5" : "#F3F4F6",
                color: m.role === "user" ? "white" : "#111",
                padding: "8px 12px", borderRadius: 12, maxWidth: "80%", fontSize: 14
              }}>
                {m.text}
              </div>
            ))}
            {loading && <div style={{ alignSelf: "flex-start", color: "#888", fontSize: 13 }}>Thinking...</div>}
            <div ref={bottomRef} />
          </div>

          {/* Suggestions */}
          {messages.length <= 1 && (
            <div style={{ padding: "4px 12px", display: "flex", flexWrap: "wrap", gap: 6 }}>
              {SUGGESTED.map(s => (
                <button key={s} onClick={() => sendMessage(s)}
                  style={{ fontSize: 12, padding: "4px 10px", borderRadius: 20,
                    border: "1px solid #4F46E5", color: "#4F46E5", background: "white", cursor: "pointer" }}>
                  {s}
                </button>
              ))}
            </div>
          )}

          {/* Input */}
          <div style={{ padding: "8px 12px", borderTop: "1px solid #eee", display: "flex", gap: 8 }}>
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === "Enter" && sendMessage()}
              placeholder="Ask about your expenses..."
              style={{ flex: 1, padding: "8px 12px", borderRadius: 20,
                border: "1px solid #ddd", fontSize: 14, outline: "none" }}
            />
            <button onClick={() => sendMessage()}
              style={{ padding: "8px 16px", background: "#4F46E5", color: "white",
                border: "none", borderRadius: 20, cursor: "pointer" }}>
              Send
            </button>
          </div>
        </div>
      )}
    </>
  );
}