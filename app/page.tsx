const project = { repository: "https://github.com/FJNeil/PolypDG-Lite", paper: "#citation" };
const contributions = [
  ["01", "True unseen-center evaluation", "A strict six-center LOCO protocol separates each test institution from training, validation, and model selection."],
  ["02", "Robustness transferred", "A consistency-trained SegFormer-B2 teacher transfers cross-center behavior to a compact SegFormer-B0 student."],
  ["03", "Deployment measured", "FP16 validation reports segmentation quality together with speed, latency, memory, power, and energy per frame."],
];
const stages = ["Selected research code", "Environment and path configuration", "Six-center LOCO protocol", "Verified result tables", "Five qualitative C4 comparisons", "Responsible-use and data notes"];

export default function Home() {
  return <main>
    <nav className="nav shell" aria-label="Primary navigation">
      <a className="wordmark" href="#top" aria-label="PolypDG-Lite home">P<span>/</span>DL</a>
      <div className="navLinks"><a href="#research">Research</a><a href="#results">Results</a><a href="#evidence">Evidence</a></div>
      <a className="navCta" href={project.repository}>GitHub +</a>
    </nav>
    <section className="hero shell" id="top">
      <div className="eyebrow"><span /> ICATI 2026 research project</div>
      <h1>PolypDG<span className="accent">-Lite</span></h1>
      <div className="heroGrid">
        <p className="lede">Cross-center polyp segmentation, distilled for low-power deployment.</p>
        <div className="heroAside"><p>A four-stage evidence chain: unseen-center diagnosis, robust teacher training, lightweight knowledge distillation, and FP16 deployment validation.</p><div className="actions"><a className="primary" href="#results">Explore results</a><a className="secondary" href={project.repository}>View code</a></div></div>
      </div>
      <div className="heroVisual" aria-label="Six-center leave-one-center-out workflow">
        <div className="domain"><b>C1-C6</b><span>Source centers</span></div><div className="signal"><i /><i /><i /></div>
        <div className="core"><small>DISTILLED MODEL</small><strong>DG</strong><em>Lite</em></div><div className="signal"><i /><i /><i /></div>
        <div className="domain domainB"><b>LOCO</b><span>Unseen center</span></div>
      </div>
      <p className="authorLine">Shih-Wei Fan Chiang and Yen-Ching Chang / Chung Shan Medical University</p>
    </section>
    <section className="statement" id="research"><div className="shell statementInner"><p className="sectionLabel">01 / Research question</p><h2>Can cross-center robustness survive <em>compression</em> into a real-time student?</h2><p className="zh">A strict leave-one-center-out study across 1,537 image-mask pairs from six clinical centers.</p></div></section>
    <section className="shell contributions">
      <div className="sectionHead"><p className="sectionLabel">02 / Core contributions</p><p className="sectionIntro">Accuracy is only one part of deployment readiness.</p></div>
      <div className="cardGrid">{contributions.map(([number,title,text]) => <article className="card" key={number}><span>{number}</span><h3>{title}</h3><p>{text}</p></article>)}</div>
    </section>
    <section className="results" id="results"><div className="shell">
      <div className="sectionHead light"><p className="sectionLabel">03 / Final FP16 student</p><h2>Results, at a glance.</h2></div>
      <div className="metricGrid"><div className="metric featured"><small>Mean Dice</small><strong>77.05</strong><span>percent across six LOCO folds</span></div><div className="metric"><small>Parameters</small><strong>3.71M</strong><span>7.08 MB estimated FP16 footprint</span></div><div className="metric"><small>Throughput</small><strong>95.26</strong><span>FPS at 10.53 ms mean latency</span></div><div className="metric"><small>Worst center</small><strong>60.44</strong><span>percent Dice on unseen C4</span></div></div>
      <div className="resultNote"><span>DEPLOYMENT PROFILE</span><p>55.43 MB peak VRAM, 14.77 W mean power, and 0.3818 J/frame. FP16 preserved the student's 77.05% Dice and 70.38% IoU.</p></div>
    </div></section>
    <section className="shell evidence" id="evidence">
      <div className="sectionHead"><p className="sectionLabel">04 / Qualitative evidence</p><p className="sectionIntro">Five difficult C4 cases, including a documented remaining failure.</p></div>
      <div className="evidenceFrame"><img src="/results/qualitative-c4-comparison.png" alt="Five C4 colonoscopy cases comparing ground truth, U-Net, robust teacher, SegFormer-B0 baseline, and the distilled student" /></div>
      <p className="caption">Columns compare image, ground truth, U-Net + LOCO, SegFormer-B2 + consistency, SegFormer-B0 + LOCO, and SegFormer-B0 + KD. Red overlays indicate predictions.</p>
    </section>
    <section className="shell reproduce"><div><p className="sectionLabel">05 / Research package</p><h2>From claim<br />to evidence.</h2><p className="bodyCopy">The repository exposes the selected training, distillation, FP16 evaluation, benchmarking, and failure-analysis scripts alongside machine-readable results.</p><a className="textLink" href={project.repository}>View source on GitHub +</a></div><ol className="checklist">{stages.map((item,index)=><li key={item}><span>{String(index+1).padStart(2,"0")}</span>{item}<b>+</b></li>)}</ol></section>
    <section className="citation" id="citation"><div className="shell"><p className="sectionLabel">06 / Paper</p><h2>PolypDG-Lite: A Deployment-Oriented Approach for Cross-Center Colonoscopic Polyp Segmentation via Domain Generalization and Low-Power Knowledge Distillation</h2><p>The 11th International Conference on Advanced Technology Innovation, 2026.</p></div></section>
    <footer><div className="shell footerInner"><div><span className="footerMark">PolypDG-Lite</span><p>Research portfolio / 2026</p></div><div className="footerLinks"><a href={project.repository}>GitHub</a><a href={project.paper}>Citation</a></div></div></footer>
  </main>;
}
