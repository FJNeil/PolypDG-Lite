const fullPaper = "/PolypDG-Lite-full-paper.pdf";

export default function Home() {
  return (
    <main className="presentationPage">
      <object
        data={`${fullPaper}#view=FitH`}
        type="application/pdf"
        aria-label="PolypDG-Lite ICATI 2026 full paper"
      >
        <div className="fallback">
          <h1>PolypDG-Lite</h1>
          <p>Your browser cannot display the full paper inline.</p>
          <a href={fullPaper}>Open the full paper PDF</a>
        </div>
      </object>
    </main>
  );
}
