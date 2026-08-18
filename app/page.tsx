const presentation = "/PolypDG-Lite-conference-presentation.pdf";

export default function Home() {
  return (
    <main className="presentationPage">
      <object
        data={`${presentation}#view=FitH`}
        type="application/pdf"
        aria-label="PolypDG-Lite ICATI 2026 conference presentation"
      >
        <div className="fallback">
          <h1>PolypDG-Lite</h1>
          <p>Your browser cannot display the presentation inline.</p>
          <a href={presentation}>Open the conference presentation PDF</a>
        </div>
      </object>
    </main>
  );
}
