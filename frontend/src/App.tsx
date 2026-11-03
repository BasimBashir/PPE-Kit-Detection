import { useEffect, type ReactNode } from "react";
import { Navigate, Route, Routes, useNavigate } from "react-router-dom";
import { auth, setUnauthorizedHandler } from "./services/api";
import { LiveProvider } from "./hooks/useLiveSocket";
import { Layout } from "./components/Layout";
import { Login } from "./features/Login";
import { Overview } from "./features/Overview";
import { CameraDetail, CameraWall } from "./features/Cameras";
import { Events } from "./features/Events";
import { CameraConfig, RulesSettings } from "./features/Admin";
import { SystemHealth } from "./features/SystemHealth";
import { DemoControl } from "./features/DemoControl";

function RequireAuth({ children, role }: { children: ReactNode; role?: "admin" }) {
  const u = auth.user();
  if (!auth.token() || !u) return <Navigate to="/login" replace />;
  if (role && u.role !== role) return <p className="error">Requires the {role} role.</p>;
  return <>{children}</>;
}

export default function App() {
  const nav = useNavigate();
  useEffect(() => setUnauthorizedHandler(() => nav("/login")), [nav]);
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route element={<RequireAuth><LiveProvider><Layout /></LiveProvider></RequireAuth>}>
        <Route index element={<Overview />} />
        <Route path="cameras" element={<CameraWall />} />
        <Route path="cameras/:id" element={<CameraDetail />} />
        <Route path="events" element={<Events />} />
        <Route path="health" element={<SystemHealth />} />
        <Route path="config/cameras" element={<RequireAuth role="admin"><CameraConfig /></RequireAuth>} />
        <Route path="config/rules" element={<RequireAuth role="admin"><RulesSettings /></RequireAuth>} />
        <Route path="demo" element={<RequireAuth role="admin"><DemoControl /></RequireAuth>} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
