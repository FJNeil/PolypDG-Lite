const project = { repository: "https://github.com/your-username/PolypDG-Lite", paper: "#results", email: "mailto:your-email@example.com" };
const contributions = [
  ["01", "Lightweight by design", "A compact research direction intended for practical deployment, with efficiency treated as a first-class objective."],
  ["02", "Generalization focused", "The evaluation is organized around robustness beyond the training distribution, not only in-domain performance."],
  ["03", "Built to be reproduced", "Code, configurations, checkpoints, and evaluation instructions are presented as one auditable research package."],
];
const checklist = ["Training and evaluation code", "Environment and dependency file", "Dataset preparation instructions", "Pretrained checkpoint and inference demo", "Full quantitative results", "Qualitative prediction gallery"];

export default function Home() {
  return <main>
    <nav className="nav shell" aria-label="Primary navigation">
      <a className="wordmark" href="#top" aria-label="PolypDG-Lite home">P<span>/</span>DL</a>
      <div className="navLinks"><a href="#research">Research</a><a href="#results">Results</a><a href="#reproduce">Reproduce</a></div>
      <a className="navCta" href={project.repository}>GitHub ↗</a>
    </nav>
    <section className="hero shell" id="top">
      <div className="eyebrow"><span /> Research project · Computer vision</div>
      <h1>PolypDG<span className="accent">—Lite</span></h1>
      <div className="heroGrid">
        <p className="lede">A lightweight domain-generalization research project for robust polyp image analysis.</p>
        <div className="heroAside"><p>Designed to study how compact models can retain reliable performance when clinical image distributions change.</p><div className="actions"><a className="primary" href="#results">Explore the results ↓</a><a className="secondary" href={project.paper}>Read the paper</a></div></div>
      </div>
      <div className="heroVisual" aria-label="Abstract visualization of model generalization">
        <div className="domain"><b>TRAIN</b><span>Source domains</span></div><div className="signal"><i /><i /><i /></div>
        <div className="core"><small>COMPACT MODEL</small><strong>DG</strong><em>Lite</em></div><div className="signal"><i /><i /><i /></div>
        <div className="domain domainB"><b>TEST</b><span>Unseen domain</span></div>
      </div>
    </section>
    <section className="statement" id="research"><div className="shell statementInner"><p className="sectionLabel">01 / Research question</p><h2>Can a smaller model generalize <em>without</em> sacrificing clinically useful detail?</h2><p className="zh">在未知資料分布下，輕量化模型能否兼顧穩健性、準確度與實際部署效率？</p></div></section>
    <section className="shell contributions">
      <div className="sectionHead"><p className="sectionLabel">02 / Core contributions</p><p className="sectionIntro">A clear account of what the project adds—and why it matters.</p></div>
      <div className="cardGrid">{contributions.map(([number,title,text]) => <article className="card" key={number}><span>{number}</span><h3>{title}</h3><p>{text}</p></article>)}</div>
    </section>
    <section className="results" id="results"><div className="shell">
      <div className="sectionHead light"><p className="sectionLabel">03 / Evidence</p><h2>Results, at a glance.</h2></div>
      <div className="metricGrid"><div className="metric featured"><small>Primary metric</small><strong>—</strong><span>Add verified mean Dice score</span></div><div className="metric"><small>Model size</small><strong>—</strong><span>Add parameter count</span></div><div className="metric"><small>Efficiency</small><strong>—</strong><span>Add FPS or FLOPs</span></div><div className="metric"><small>Generalization</small><strong>—</strong><span>Add improvement vs. baseline</span></div></div>
      <div className="resultNote"><span>DATA INTEGRITY NOTE</span><p>Metrics are intentionally blank until verified experimental outputs are added. No result on this page is fabricated.</p></div>
    </div></section>
    <section className="shell reproduce" id="reproduce"><div><p className="sectionLabel">04 / Reproducibility</p><h2>From repository<br />to result.</h2><p className="bodyCopy">The final repository should let reviewers understand the work quickly and reproduce the central experiment with minimal guesswork.</p><a className="textLink" href={project.repository}>View source on GitHub ↗</a></div><ol className="checklist">{checklist.map((item,index)=><li key={item}><span>{String(index+1).padStart(2,"0")}</span>{item}<b>○</b></li>)}</ol></section>
    <footer><div className="shell footerInner"><div><span className="footerMark">PolypDG—Lite</span><p>Research portfolio · 2026</p></div><div className="footerLinks"><a href={project.repository}>GitHub</a><a href={project.paper}>Paper</a><a href={project.email}>Contact</a></div></div></footer>
  </main>;
}
