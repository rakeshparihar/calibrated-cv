// Records the API's real docs page (FastAPI's Swagger UI) running a scripted sequence of calls.
// Usage: node scripts/record_docs.mjs scripts/demo_steps.json   (API on BASE_URL, default http://localhost:8000)
// Each step: {"op": "POST /path", "params": {"name": "value" | ["a", "b"]}, "body": {...}, "file": "path",
//             "save": {"var": "field.in.response"}}. Strings may use {{var}} from earlier saves.
// Output: docs/img/api.webm and docs/img/api.png
import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';

const { chromium } = createRequire(import.meta.url)('playwright');
const steps = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const base = process.env.BASE_URL || 'http://localhost:8000';
const vars = {};
const fill = (v) => (typeof v === 'string' ? v.replace(/\{\{(\w+)\}\}/g, (_, k) => vars[k]) : v);
const deep = (o) => JSON.parse(JSON.stringify(o), (_, v) => fill(v));
const pick = (obj, dotted) => dotted.split('.').reduce((o, k) => o?.[k], obj);

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1100, height: 800 }, recordVideo: { dir: 'docs/img/.rec', size: { width: 1100, height: 800 } } });
const page = await ctx.newPage();
await page.goto(`${base}/docs`);
await page.waitForSelector('.opblock');
await page.waitForTimeout(1500);

let open = null;
for (const step of steps) {
  const [method, route] = step.op.split(' ');
  const op = page.locator(`.opblock.opblock-${method.toLowerCase()}`).filter({ has: page.locator(`.opblock-summary-path[data-path="${route}"]`) }).first();
  if (open && open !== step.op) { // collapse the previous call so the page stays readable
    const prev = page.locator('.opblock.is-open .opblock-summary').first();
    if (await prev.count()) await prev.click();
  }
  if (open !== step.op) {
    await op.locator('.opblock-summary').scrollIntoViewIfNeeded();
    await op.locator('.opblock-summary').click();
    await op.getByRole('button', { name: 'Try it out' }).click();
    open = step.op;
  }
  for (const [name, value] of Object.entries(step.params || {})) {
    const row = op.locator(`tr[data-param-name="${name}"]`);
    if (Array.isArray(value)) {
      for (const [i, item] of value.entries()) {
        if ((await row.locator('.json-schema-form-item input').count()) <= i) await row.getByRole('button', { name: /Add/ }).click();
        await row.locator('.json-schema-form-item input').nth(i).fill(String(fill(item)));
      }
    } else {
      await row.locator('input').first().fill(String(fill(value)));
    }
  }
  if (step.body) await op.locator('textarea.body-param__text').fill(JSON.stringify(deep(step.body), null, 2));
  if (step.file) await op.locator('input[type=file]').setInputFiles(path.resolve(step.file));
  await page.waitForTimeout(700);
  await op.getByRole('button', { name: 'Execute' }).click();
  const out = op.locator('.live-responses-table .response-col_description .microlight').first();
  await out.waitFor();
  await page.waitForTimeout(300);
  const text = await out.innerText();
  await op.locator('.live-responses-table').scrollIntoViewIfNeeded();
  await out.evaluate((el) => el.scrollIntoView({ block: 'center' }));
  for (const [k, field] of Object.entries(step.save || {})) vars[k] = pick(JSON.parse(text), field);
  await page.waitForTimeout(step.pause ?? 2600);
}
await page.screenshot({ path: 'docs/img/api.png' });
const v = page.video();
await ctx.close();
fs.renameSync(await v.path(), 'docs/img/api.webm');
fs.rmSync('docs/img/.rec', { recursive: true, force: true });
await browser.close();
