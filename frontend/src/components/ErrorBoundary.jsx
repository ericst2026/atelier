import React from "react";

/** A page that throws should say what happened rather than going white — in a
 *  classroom nobody is going to open the browser console. */
export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error("page failed:", error, info);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <main className="page">
        <div className="panel stack" style={{ borderColor: "var(--dup)" }}>
          <h2>This page stopped</h2>
          <p className="muted">Something in the page threw an error. The rest of the app is still fine.</p>
          <pre className="log" style={{ maxHeight: 220 }}>
            {String(this.state.error?.stack || this.state.error)}
          </pre>
          <div className="row">
            <button className="btn primary" onClick={() => this.setState({ error: null })}>
              Try again
            </button>
            <button className="btn" onClick={() => window.location.assign("/")}>
              Back to the experiments
            </button>
          </div>
        </div>
      </main>
    );
  }
}
