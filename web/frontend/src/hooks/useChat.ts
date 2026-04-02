import { useCallback, useState } from "react";
import { sendChat } from "../api/client";
import type { ChatMessage } from "../types";

export function useChat(sessionId: string | null, selectedSlices: Set<number>) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const send = useCallback(
    async (text: string) => {
      if (!sessionId || isLoading) return;

      const userMsg: ChatMessage = { role: "user", content: text };
      const updatedHistory = [...messages, userMsg];
      setMessages(updatedHistory);
      setIsLoading(true);

      try {
        const res = await sendChat(
          sessionId,
          text,
          Array.from(selectedSlices).sort((a, b) => a - b),
          // Send prior history (without the current message, which is sent as `message`)
          messages
        );

        console.log("Chat response:", JSON.stringify(res));
        const content = res.response || "(Empty response from model)";
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content },
        ]);
      } catch (err) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `Error: ${err instanceof Error ? err.message : "Unknown error"}`,
          },
        ]);
      } finally {
        setIsLoading(false);
      }
    },
    [sessionId, selectedSlices, messages, isLoading]
  );

  const reset = useCallback(() => {
    setMessages([]);
  }, []);

  return { messages, isLoading, send, reset };
}
