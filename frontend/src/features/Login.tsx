import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { api, auth } from "../services/api";

export function Login() {
  const [username, setU] = useState("admin");
  const [password, setP] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const nav = useNavigate();
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      const r = await api.login(username, password);
      auth.set(r.access_token, r.user);
      nav("/");
    } catch (x) {
      setErr((x as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <main className="login">
      <form onSubmit={submit} className="card login-card" aria-labelledby="login-title">
        <h1 id="login-title"><span className="brand-mark" aria-hidden="true">◆</span> PPE Intelligence</h1>
        <p className="muted">Offline hard-hat compliance monitoring · independent portfolio demo</p>
        <label>Username<input value={username} onChange={(e) => setU(e.target.value)} autoComplete="username" required /></label>
        <label>Password<input type="password" value={password} onChange={(e) => setP(e.target.value)} autoComplete="current-password" required /></label>
        {err && <p className="error" role="alert">{err}</p>}
        <button className="btn primary" disabled={busy}>{busy ? "Signing in…" : "Sign in"}</button>
      </form>
    </main>
  );
}
