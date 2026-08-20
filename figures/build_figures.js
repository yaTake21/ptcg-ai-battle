#!/usr/bin/env node
/**
 * build_figures.js — README 用の図を生成する。
 *
 * 1つのレイアウト定義から2つの成果物を出力する（内容が食い違わないようにするため）:
 *   architecture.pptx  … 編集用（PowerPoint / Keynote / Google スライドで開ける）
 *   fig1_network.svg   … README 表示用（スライド1）
 *   fig2_training.svg  … README 表示用（スライド2）
 *
 * 使い方:  NODE_PATH=<pptxgenjs のある node_modules> node figures/build_figures.js
 *
 * 方針: 図の中に説明文は書かない（説明は README 本文で行う）。ラベルは英語。
 *       処理は左から右へ流す。白箱＋細い枠線／水色のグループ枠／灰色のブロック矢印／
 *       楕円＝出力、というスタイルで統一する。
 */
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const OUT = __dirname;
const W = 13.333, H = 7.5, DPI = 96;

const C = {
  ink: "333333", sub: "767676", border: "808080",
  boxFill: "FFFFFF", grpFill: "DEEBF7", grpBorder: "4472C4", grpTitle: "2E5C9A",
  block: "A6A6A6", arrow: "404040", dash: "9AA0A6", white: "FFFFFF",
};
const FONT = "Arial";
const SVG_FONT = "Arial, Helvetica, sans-serif";

// shape: {kind:'rect'|'round'|'ellipse'|'group'|'block', x,y,w,h, lines, size, bold, color,
//         fill, border, dashed, sub, subAlign, title}
// arrow: {x1,y1,x2,y2, dashed}
// text : {x,y,w, lines, size, bold, color, align}


// ============================ 図0: デッキ選定・更新ループ ============================
const f0 = { title: "Deck selection and update loop", crop: [0.25, 4.60], shapes: [], arrows: [], texts: [] };
const CY0 = 2.05;

f0.shapes.push({ kind: "rect", x: 0.40, y: CY0 - 0.42, w: 1.55, h: 0.84, lines: ["Leaderboard", "decks"], size: 11 });
f0.arrows.push({ x1: 1.95, y1: CY0, x2: 2.30, y2: CY0 });
f0.shapes.push({ kind: "round", x: 2.30, y: CY0 - 0.42, w: 1.80, h: 0.84,
  lines: ["Classify archetypes,", "score-band share"], size: 10 });
f0.arrows.push({ x1: 3.20, y1: CY0 + 0.42, x2: 3.20, y2: CY0 + 1.02 });
f0.shapes.push({ kind: "rect", x: 2.30, y: CY0 + 1.02, w: 1.80, h: 0.70,
  lines: ["Design opponent pool"], size: 10 });
f0.arrows.push({ x1: 4.10, y1: CY0, x2: 4.45, y2: CY0 });
f0.shapes.push({ kind: "round", x: 4.45, y: CY0 - 0.42, w: 1.80, h: 0.84,
  lines: ["Collect match history", "of top submissions"], size: 10 });
f0.arrows.push({ x1: 6.25, y1: CY0, x2: 6.60, y2: CY0 });
f0.shapes.push({ kind: "round", x: 6.60, y: CY0 - 0.42, w: 1.85, h: 0.84,
  lines: ["Identify the 60 cards,", "group into variants"], size: 10 });
f0.arrows.push({ x1: 8.45, y1: CY0, x2: 8.80, y2: CY0 });
f0.shapes.push({ kind: "round", x: 8.80, y: CY0 - 0.42, w: 1.70, h: 0.84,
  lines: ["Top players", "switched variants?"], size: 10 });
f0.arrows.push({ x1: 10.50, y1: CY0, x2: 10.85, y2: CY0 });
f0.texts.push({ x: 10.48, y: CY0 - 0.38, w: 0.5, size: 9, color: C.sub, lines: ["Yes"] });
f0.shapes.push({ kind: "round", x: 10.85, y: CY0 - 0.42, w: 2.00, h: 0.84,
  lines: ["Collect new-variant replays,", "retrain, update the deck"], size: 9.5 });
f0.arrows.push({ x1: 11.85, y1: CY0 + 0.42, x2: 11.85, y2: CY0 + 1.02 });
f0.shapes.push({ kind: "ellipse", x: 10.85, y: CY0 + 1.02, w: 2.00, h: 0.75,
  lines: ["Verify by direct play"], size: 10 });
f0.arrows.push({ x1: 9.65, y1: CY0 + 0.42, x2: 9.65, y2: CY0 + 1.40 });
f0.texts.push({ x: 9.70, y: CY0 + 0.50, w: 0.9, size: 9, color: C.sub, align: "left", lines: ["No: keep"] });
f0.arrows.push({ dashed: true, points: [[10.85, CY0 + 1.40], [5.35, CY0 + 1.40], [5.35, CY0 + 0.42]] });

// ============================ 図1: ネットワーク ============================
const f1 = { title: "Network — one decision", crop: [0.25, 3.80], shapes: [], arrows: [], texts: [] };
const CY = 2.20;   // 主流のy中心

["Board state", "Legal options", "Turn context"].forEach((t, i) => {
  f1.shapes.push({ kind: "rect", x: 0.40, y: 1.18 + i * 0.72, w: 1.35, h: 0.6, lines: [t], size: 11 });
});
f1.shapes.push({ kind: "block", x: 1.92, y: CY - 0.21, w: 0.42, h: 0.42 });

f1.shapes.push({ kind: "group", x: 2.48, y: 1.28, w: 3.90, h: 1.85, title: "Input encoding",
  sub: "concat 404 dims → Linear 404→256" });
[["ID", "embeddings"], ["Static", "attributes"], ["Numeric", "features"]].forEach((ls, i) => {
  f1.shapes.push({ kind: "round", x: 2.64 + i * 1.22, y: 1.92, w: 1.14, h: 0.82, lines: ls, size: 10 });
});
f1.shapes.push({ kind: "block", x: 6.55, y: CY - 0.21, w: 0.42, h: 0.42 });

f1.shapes.push({ kind: "round", x: 7.12, y: CY - 0.41, w: 1.70, h: 0.82,
  lines: ["Transformer", "Encoder"], size: 11, sub: "x6 layers, d=256, 8 heads" });
f1.arrows.push({ x1: 8.82, y1: CY, x2: 9.18, y2: CY - 0.34 });
f1.shapes.push({ kind: "round", x: 9.18, y: 1.45, w: 1.42, h: 0.72, lines: ["Policy Head"], size: 10.5,
  sub: "MLP 256→64→1" });
f1.arrows.push({ x1: 8.82, y1: CY, x2: 9.18, y2: CY + 0.66, dashed: true });
f1.shapes.push({ kind: "round", x: 9.18, y: 2.56, w: 1.42, h: 0.72, lines: ["Value Head"], size: 10.5,
  dashed: true, color: C.sub, sub: "not used" });

f1.arrows.push({ x1: 10.60, y1: 1.81, x2: 10.98, y2: 1.81 });
f1.shapes.push({ kind: "ellipse", x: 10.98, y: 1.39, w: 1.95, h: 0.84, lines: ["Selected action"], size: 11,
  sub: "argmax over K options" });

// ============================ 図2: 学習パイプライン ============================
const f2 = { title: "Training pipeline", crop: [0.25, 4.45], shapes: [], arrows: [], texts: [] };
const CY2 = 2.60;

f2.shapes.push({ kind: "rect", x: 0.40, y: CY2 - 0.42, w: 1.35, h: 0.84, lines: ["Top-player", "replays"], size: 11 });
f2.shapes.push({ kind: "block", x: 1.90, y: CY2 - 0.21, w: 0.42, h: 0.42 });

f2.shapes.push({ kind: "round", x: 2.45, y: CY2 - 0.42, w: 1.60, h: 0.84, lines: ["Feature", "encoding"], size: 11 });
f2.shapes.push({ kind: "block", x: 4.20, y: CY2 - 0.21, w: 0.42, h: 0.42 });

[{ label: "seed 42", y: 1.35 }, { label: "seed 0", y: 3.10 }].forEach((r) => {
  f2.texts.push({ x: 4.80, y: r.y - 0.28, w: 1.40, size: 9.5, color: C.sub, align: "left", lines: [r.label] });
  f2.shapes.push({ kind: "round", x: 4.80, y: r.y, w: 1.40, h: 0.75, lines: ["Stage 1"], size: 11, sub: "14,459 games" });
  f2.arrows.push({ x1: 6.20, y1: r.y + 0.375, x2: 6.55, y2: r.y + 0.375 });
  f2.shapes.push({ kind: "round", x: 6.55, y: r.y, w: 1.40, h: 0.75, lines: ["Stage 2"], size: 11, sub: "6,569 games" });
  f2.arrows.push({ x1: 7.95, y1: r.y + 0.375, x2: 8.30, y2: r.y + 0.375 });
  f2.shapes.push({ kind: "round", x: 8.30, y: r.y, w: 1.30, h: 0.75, lines: ["WiSE-FT"], size: 11, sub: "α = 0.5" });
  f2.arrows.push({ x1: 9.60, y1: r.y + 0.375, x2: 10.15, y2: r.y < 2 ? CY2 - 0.30 : CY2 + 0.30 });
});
f2.shapes.push({ kind: "ellipse", x: 10.15, y: CY2 - 0.42, w: 1.50, h: 0.84, lines: ["Ensemble"], size: 11,
  sub: "probability average" });
f2.arrows.push({ x1: 11.65, y1: CY2, x2: 11.95, y2: CY2 });
f2.shapes.push({ kind: "ellipse", x: 11.95, y: CY2 - 0.42, w: 1.15, h: 0.84, lines: ["Submission"], size: 11 });

const FIGS = [f0, f1, f2];

// ============================ PPTX ============================
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";

FIGS.forEach((f) => {
  const sl = pres.addSlide();
  sl.background = { color: C.white };
  // SVG は使用範囲だけを切り出すが、pptx はスライド全面なので内容を縦中央に寄せる
  const [q0, q1] = f.crop || [0, H];
  const yo = (H - (q1 - q0)) / 2 - q0;
  sl.addText(f.title, { x: 0.4, y: 0.45 + yo, w: 12.5, h: 0.5, fontSize: 20, bold: true, color: C.ink, fontFace: FONT, margin: 0 });

  f.shapes.filter((s) => s.kind === "group").forEach((s) => {
    sl.addShape(pres.ShapeType.rect, { x: s.x, y: s.y + yo, w: s.w, h: s.h,
      fill: { color: C.grpFill }, line: { color: C.grpBorder, width: 1 } });
    sl.addText(s.title, { x: s.x + 0.14, y: s.y + yo + 0.12, w: s.w - 0.28, h: 0.32, fontSize: 12,
      color: C.grpTitle, fontFace: FONT, align: "left", margin: 0 });
    if (s.sub) sl.addText(s.sub, { x: s.x, y: s.y + yo + s.h + 0.05, w: s.w, h: 0.24, fontSize: 9,
      color: C.sub, fontFace: FONT, align: "center", valign: "top", margin: 0 });
  });
  f.arrows.forEach((a) => {
    const col = a.dashed ? C.dash : C.arrow;
    const dt = a.dashed ? "dash" : "solid";
    if (a.points) {                       // 折れ線: 最終区間にだけ矢尻を付ける
      for (let i = 0; i + 1 < a.points.length; i++) {
        const [x1, y1] = a.points[i], [x2, y2] = a.points[i + 1];
        const last = i + 2 === a.points.length;
        sl.addShape(pres.ShapeType.line, { x: x1, y: y1 + yo, w: x2 - x1, h: y2 - y1,
          line: { color: col, width: 1.25, dashType: dt,
                  endArrowType: last ? "triangle" : "none" } });
      }
      return;
    }
    sl.addShape(pres.ShapeType.line, { x: a.x1, y: a.y1 + yo, w: a.x2 - a.x1, h: a.y2 - a.y1,
      line: { color: col, width: 1.25, endArrowType: "triangle", dashType: dt } });
  });
  f.shapes.filter((s) => s.kind !== "group").forEach((s) => {
    const st = s.kind === "block" ? pres.ShapeType.rightArrow
      : s.kind === "ellipse" ? pres.ShapeType.ellipse
      : s.kind === "round" ? pres.ShapeType.roundRect : pres.ShapeType.rect;
    sl.addShape(st, { x: s.x, y: s.y + yo, w: s.w, h: s.h,
      fill: { color: s.kind === "block" ? C.block : (s.fill || C.boxFill) },
      rectRadius: s.kind === "round" ? 0.08 : undefined,
      line: s.kind === "block" ? { color: C.block, width: 0 }
        : { color: s.dashed ? C.dash : C.border, width: 1, dashType: s.dashed ? "dash" : "solid" } });
    if (s.lines) {
      sl.addText(s.lines.join("\n"), { x: s.x + 0.04, y: s.y + yo, w: s.w - 0.08, h: s.h,
        fontSize: s.size || 11, bold: !!s.bold, color: s.color || C.ink, fontFace: FONT,
        align: "center", valign: "middle", margin: 0, lineSpacingMultiple: 1.05 });
    }
    if (s.sub) {
      sl.addText(s.sub, { x: s.x - 0.35, y: s.y + yo + s.h + 0.04, w: s.w + 0.7, h: 0.24,
        fontSize: 9, color: C.sub, fontFace: FONT, align: "center", valign: "top", margin: 0 });
    }
  });
  f.texts.forEach((t) => {
    sl.addText(t.lines.join("\n"), { x: t.x, y: t.y + yo, w: t.w, h: 0.24 * t.lines.length,
      fontSize: t.size || 10, bold: !!t.bold, color: t.color || C.ink, fontFace: FONT,
      align: t.align || "center", valign: "top", margin: 0 });
  });
});

// ============================ SVG ============================
const px = (v) => Math.round(v * DPI * 100) / 100;
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

function svgText(lines, x, y, w, h, size, bold, color, align, valign) {
  const fs2 = size * 1.33;
  const anchor = align === "left" ? "start" : "middle";
  const tx = align === "left" ? px(x) : px(x + w / 2);
  const lh = fs2 * 1.25;
  const ty = valign === "middle" ? px(y + h / 2) - (lines.length * lh) / 2 + lh * 0.78
                                 : px(y) + lh * 0.82;
  return lines.map((ln, i) =>
    `<text x="${tx}" y="${ty + i * lh}" font-family="${SVG_FONT}" font-size="${fs2}" ` +
    `font-weight="${bold ? 700 : 400}" fill="#${color}" text-anchor="${anchor}">${esc(ln)}</text>`).join("\n");
}

FIGS.forEach((f, idx) => {
  const p = [];
  const [cy0, cy1] = f.crop || [0, H];   // SVG は使っている範囲だけを切り出す（横長になる）
  p.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 ${px(cy0)} ${px(W)} ${px(cy1 - cy0)}" ` +
    `width="${px(W)}" height="${px(cy1 - cy0)}" role="img">`);
  p.push(`<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">` +
    `<path d="M 0 0 L 10 5 L 0 10 z" fill="#${C.arrow}"/></marker>` +
    `<marker id="ad" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">` +
    `<path d="M 0 0 L 10 5 L 0 10 z" fill="#${C.dash}"/></marker></defs>`);
  p.push(`<rect x="0" y="${px(cy0)}" width="${px(W)}" height="${px(cy1 - cy0)}" fill="#${C.white}"/>`);
  p.push(svgText([f.title], 0.4, 0.45, 12.5, 0.5, 20, true, C.ink, "left", "top"));

  f.shapes.filter((s) => s.kind === "group").forEach((s) => {
    p.push(`<rect x="${px(s.x)}" y="${px(s.y)}" width="${px(s.w)}" height="${px(s.h)}" fill="#${C.grpFill}" stroke="#${C.grpBorder}" stroke-width="1.2"/>`);
    p.push(svgText([s.title], s.x + 0.14, s.y + 0.12, s.w, 0.3, 12, false, C.grpTitle, "left", "top"));
    if (s.sub) p.push(svgText([s.sub], s.x, s.y + s.h + 0.05, s.w, 0.24, 9, false, C.sub, "center", "top"));
  });
  f.arrows.forEach((a) => {
    const col = a.dashed ? C.dash : C.arrow;
    const dash = a.dashed ? ' stroke-dasharray="5 4"' : "";
    const mk = `marker-end="url(#${a.dashed ? "ad" : "a"})"`;
    if (a.points) {                       // 折れ線は polyline 1本で描く
      const pts = a.points.map(([x, y]) => `${px(x)},${px(y)}`).join(" ");
      p.push(`<polyline points="${pts}" fill="none" stroke="#${col}" stroke-width="1.6"${dash} ${mk}/>`);
      return;
    }
    p.push(`<line x1="${px(a.x1)}" y1="${px(a.y1)}" x2="${px(a.x2)}" y2="${px(a.y2)}" ` +
      `stroke="#${col}" stroke-width="1.6"${dash} ${mk}/>`);
  });
  f.shapes.filter((s) => s.kind !== "group").forEach((s) => {
    if (s.kind === "block") {
      const x0 = px(s.x), y0 = px(s.y), w0 = px(s.w), h0 = px(s.h);
      const tail = h0 * 0.28, head = w0 * 0.45;
      p.push(`<polygon points="${x0},${y0 + tail} ${x0 + w0 - head},${y0 + tail} ${x0 + w0 - head},${y0} ` +
        `${x0 + w0},${y0 + h0 / 2} ${x0 + w0 - head},${y0 + h0} ${x0 + w0 - head},${y0 + h0 - tail} ${x0},${y0 + h0 - tail}" fill="#${C.block}"/>`);
      return;
    }
    const stroke = s.dashed ? C.dash : C.border;
    if (s.kind === "ellipse") {
      p.push(`<ellipse cx="${px(s.x + s.w / 2)}" cy="${px(s.y + s.h / 2)}" rx="${px(s.w / 2)}" ry="${px(s.h / 2)}" ` +
        `fill="#${s.fill || C.boxFill}" stroke="#${stroke}" stroke-width="1.2"/>`);
    } else {
      p.push(`<rect x="${px(s.x)}" y="${px(s.y)}" width="${px(s.w)}" height="${px(s.h)}" rx="${s.kind === "round" ? 8 : 0}" ` +
        `fill="#${s.fill || C.boxFill}" stroke="#${stroke}" stroke-width="1.2"${s.dashed ? ' stroke-dasharray="5 4"' : ""}/>`);
    }
    if (s.lines) p.push(svgText(s.lines, s.x + 0.04, s.y, s.w - 0.08, s.h, s.size || 11, !!s.bold, s.color || C.ink, "center", "middle"));
    if (s.sub) p.push(svgText([s.sub], s.x - 0.35, s.y + s.h + 0.04, s.w + 0.7, 0.24, 9, false, C.sub, "center", "top"));
  });
  f.texts.forEach((t) => {
    p.push(svgText(t.lines, t.x, t.y, t.w, 0.24, t.size || 10, !!t.bold, t.color || C.ink, t.align || "center", "top"));
  });
  p.push("</svg>");
  fs.writeFileSync(path.join(OUT, ["fig0_loop.svg", "fig1_network.svg", "fig2_training.svg"][idx]), p.join("\n"));
});

pres.writeFile({ fileName: path.join(OUT, "architecture.pptx") })
  .then(() => console.log("wrote architecture.pptx / fig1_network.svg / fig2_training.svg"));
