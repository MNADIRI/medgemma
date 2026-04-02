import { useCallback, useState } from "react";
import { sendChat } from "../api/client";
import type { ChatMessage, ROI } from "../types";

export function useChat(
  sessionId: string | null,
  selectedSlices: Set<number>,
  roiMap: Map<number, ROI>
) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const send = useCallback(
    async (text: string) => {
      if (!sessionId || isLoading) return;

      const userMsg: ChatMessage = { role: "user", content: text };
      const updatedHistory = [...messages, userMsg];
      setMessages(updatedHistory);
      setIsLoading(true);

      // Build ROIs payload: only include ROIs for selected slices
      const sliceIndices = Array.from(selectedSlices).sort((a, b) => a - b);
      const rois: Record<string, ROI> = {};
      for (const idx of sliceIndices) {
        const roi = roiMap.get(idx);
        if (roi) rois[String(idx)] = roi;
      }

      try {
        const res = await sendChat(
          sessionId,
          text,
          sliceIndices,
          messages,
          rois
        );

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
    [sessionId, selectedSlices, roiMap, messages, isLoading]
  );

  const reset = useCallback(() => {
    setMessages([]);
  }, []);

  return { messages, isLoading, send, reset };
}
