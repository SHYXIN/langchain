import { useState, useCallback, useRef } from "react";
import { Header } from "@/components/Header";
import { SessionList } from "@/components/SessionList";
import { ChatWindow } from "@/components/ChatWindow";
import { useSessions } from "@/hooks/useSessions";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  references?: Array<{
    content: string;
    response: string;
    category: string;
  }>;
}

function App() {
  const {
    sessions,
    activeSession,
    activeId,
    createSession,
    deleteSession,
    switchSession,
    updateMessages,
    renameSession,
  } = useSessions();

  const [sidebarOpen, setSidebarOpen] = useState(true);
  // 使用 ref 保存 sessionId，避免组件重新挂载时丢失
  const sessionIdRef = useRef(`session_${Date.now()}`);

  const handleAddMessage = useCallback(
    (message: Message) => {
      if (activeId) {
        const currentMessages = activeSession?.messages || [];
        updateMessages(activeId, [...currentMessages, message]);
      }
    },
    [activeId, activeSession, updateMessages]
  );

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* 左侧会话列表 */}
      {sidebarOpen && (
        <SessionList
          sessions={sessions.map((s) => ({
            id: s.id,
            title: s.title,
            updatedAt: s.updatedAt,
          }))}
          currentSessionId={activeId || ""}
          onSelectSession={switchSession}
          onCreateSession={createSession}
          onDeleteSession={deleteSession}
          onRenameSession={renameSession}
        />
      )}

      {/* 右侧聊天区域 */}
      <div className="flex flex-1 flex-col overflow-hidden">
        <Header onToggleSidebar={() => setSidebarOpen((p) => !p)} />
        <main className="flex-1 overflow-hidden">
          {activeSession ? (
            <ChatWindow
              sessionId={sessionIdRef.current}
              messages={activeSession.messages}
              onAddMessage={handleAddMessage}
            />
          ) : (
            <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
              请选择一个会话或新建会话开始对话
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
