import React from "react";
import { Navigate, Outlet, Route, Routes, useLocation } from "react-router-dom";
import ErrorBoundary from "./components/ErrorBoundary";
import TopBar from "./components/TopBar";
import { useAuth } from "./lib/auth";
import Account from "./pages/Account";
import Admin from "./pages/Admin";
import Display from "./pages/Display";
import Experiment from "./pages/Experiment";
import Home from "./pages/Home";
import Login from "./pages/Login";
import SignUp from "./pages/SignUp";
import SubmissionReview from "./pages/SubmissionReview";
import Teacher from "./pages/Teacher";

function RequireAuth({ teacher = false, admin = false }) {
  const { user, ready, isTeacher, isAdmin } = useAuth();
  const loc = useLocation();
  if (!ready) return <div className="page muted">Loading…</div>;
  if (!user) return <Navigate to="/login" state={{ from: loc.pathname }} replace />;
  if (teacher && !isTeacher) return <Navigate to="/" replace />;
  if (admin && !isAdmin) return <Navigate to="/" replace />;
  return (
    <div className="app">
      <TopBar />
      <ErrorBoundary>
        <Outlet />
      </ErrorBoundary>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/signup" element={<SignUp />} />
      <Route path="/display/:n" element={<Display />} />
      <Route element={<RequireAuth />}>
        <Route path="/" element={<Home />} />
        <Route path="/account" element={<Account />} />
        <Route path="/experiments/:slug" element={<Experiment />} />
      </Route>
      <Route element={<RequireAuth admin />}>
        <Route path="/admin" element={<Admin />} />
      </Route>
      <Route element={<RequireAuth teacher />}>
        <Route path="/teacher" element={<Teacher />} />
        <Route path="/teacher/submissions/:id" element={<SubmissionReview />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
