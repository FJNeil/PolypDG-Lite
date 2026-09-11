import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const workerUrl = new URL("../dist/server/index.js", import.meta.url);

async function render() {
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the PolypDG-Lite conference presentation", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>PolypDG-Lite \| Conference Presentation<\/title>/i);
  assert.match(html, /PolypDG-Lite-conference-presentation\.pdf/i);
  assert.match(html, /type="application\/pdf"/i);
  assert.match(html, /Cross-center robustness\. Low-power deployment\./i);
  assert.doesNotMatch(
    html,
    /Your site is taking shape|Building your site|Starter Project|codex-preview|skeleton/i,
  );
});

test("repository metadata no longer identifies the starter template", async () => {
  const [packageJson, workerSource] = await Promise.all([
    readFile(new URL("../package.json", import.meta.url), "utf8"),
    readFile(new URL("../worker/index.ts", import.meta.url), "utf8"),
  ]);

  assert.match(packageJson, /"name":\s*"polypdg-lite-research-showcase"/);
  assert.doesNotMatch(packageJson, /site-creator-vinext-starter/);
  assert.doesNotMatch(workerSource, /vinext-starter template/i);
});
