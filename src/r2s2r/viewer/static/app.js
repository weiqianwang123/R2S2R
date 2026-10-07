// R2S2R run viewer: a progress axis over the stages; the recording, the chosen frames,
// the support, the objects and their physical parameters open on a click. Articulated
// objects get a slider per joint wherever they are shown in 3D.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

const POLL_MS = 5000;
const NODES = [
  { id: 'capture', label: 'Capture', stage: null, view: true, title: 'Capture: every camera, and the robot along the recording' },
  { id: 'frames', label: 'Frames', stage: '2', view: true, title: 'Frames chosen to reconstruct from' },
  { id: 'support', label: 'Support', stage: '2', view: true, title: 'Support surface' },
  { id: 'objects', label: 'Objects', stage: '3', view: true, title: 'Objects' },
  { id: 'physics', label: 'Physics', stage: '4', view: true, title: 'Physical parameters' },
  { id: 'settle', label: 'Settle', stage: '5', view: false },
  { id: 'refine', label: 'Refine', stage: '6', view: false },
];
const ROLE_COLORS = ['#5aa9ff', '#f0b429', '#3fb97a', '#ef5b5b', '#b08cff'];

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const RUN = new URLSearchParams(location.search).get('run') || '';
const Q = `run=${encodeURIComponent(RUN)}`;
const enc = (p) => p.split('/').map(encodeURIComponent).join('/');
const fileUrl = (p) => `/f/${enc(p)}?${Q}`;
const thumbUrl = (p, w = 480) => `/thumb/${enc(p)}?w=${w}&${Q}`;
const inRun = (p) => p && !p.startsWith('/');
// Turns a mesh file's up axis to z (as tools/geometry.py UP_ROTATIONS does).
const UP_EULER = { y: [Math.PI / 2, 0, 0], '-y': [-Math.PI / 2, 0, 0], x: [0, -Math.PI / 2, 0], z: [0, 0, 0] };
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(+v) ? '—' : Number(v).toFixed(d));
const iouClass = (v) => (v >= 0.85 ? 'iou-good' : v >= 0.7 ? 'iou-mid' : 'iou-bad');
// A joint position as people read it: degrees for a turn, centimetres for a slide.
const jointValue = (type, v) => (type === 'prismatic' ? `${fmt(v * 100, 1)} cm` : `${fmt(v * 180 / Math.PI, 0)}°`);

const S = {
  rec: null, robot: null, state: null, open: null, panelKey: '', selected: null,
  idx: 0, t: 0, playing: false, speed: 1,
  framesByRole: {}, frameById: {}, roleColor: {}, chosen: new Set(), replayPanels: {},
  scenePath: null, sceneUserPicked: false, captureReady: false,
};

// ------------------------------------------------------------------------ start
async function init() {
  S.rec = await (await fetch(`/api/recording?${Q}`)).json();
  listRuns();
  $('capture-name').textContent = S.rec.name;
  $('instruction').textContent = S.rec.instruction ? `“${S.rec.instruction}”` : '';
  document.title = `R2S2R Run Viewer · ${S.rec.name}`;
  S.rec.cameras.forEach((c, k) => { S.roleColor[c.role] = ROLE_COLORS[k % ROLE_COLORS.length]; });
  for (const f of S.rec.frames) {
    (S.framesByRole[f.camera] ||= []).push(f);
    S.frameById[f.id] = f;
  }
  bindUi();
  await poll();
  openPanel('capture'); // the recording is what the page opens on
  setInterval(poll, POLL_MS);
}

async function listRuns() {
  const names = await (await fetch('/api/runs')).json();
  const sel = $('run-select');
  sel.innerHTML = names.map((n) => `<option value="${esc(n)}">${esc(n)}</option>`).join('');
  sel.value = RUN || names[0] || '';
  sel.onchange = () => { location.search = `?run=${encodeURIComponent(sel.value)}`; };
}

function bindUi() {
  $('axis').addEventListener('click', (e) => {
    const node = e.target.closest('.node.viewable');
    if (node) openPanel(node.dataset.id === S.open ? null : node.dataset.id);
  });
  $('panel-close').addEventListener('click', () => openPanel(null));
  $('panel-body').addEventListener('click', onPanelClick);
  $('lb-close').addEventListener('click', closeLightbox);
  document.addEventListener('keydown', (e) => {
    if (e.code === 'Escape') { closeLightbox(); return; }
    if (S.open !== 'capture' || (e.target.tagName === 'INPUT' && e.target.type !== 'range')) return;
    if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
    if (e.code === 'ArrowRight') setIdx(S.idx + 1);
    if (e.code === 'ArrowLeft') setIdx(S.idx - 1);
  });
}

// ---------------------------------------------------------------------- polling
async function poll() {
  let state;
  try {
    state = await (await fetch(`/api/state?${Q}`)).json();
  } catch (e) {
    $('activity').textContent = 'server unreachable';
    return;
  }
  S.state = state;
  S.chosen = new Set((state.frames || []).map((f) => f.id));
  S.replayPanels = {};
  for (const p of state.replay_panels || []) {
    const m = p.match(/(\d+)_([^/]+)\.png$/);
    if (m) S.replayPanels[`${m[2]}@${+m[1]}`] = p;
  }
  renderAxis();
  if (S.open) renderPanel(false);
  if (S.captureReady && S.robot) {
    updateSceneSelect(state.scenes);
    $('replay-toggle').classList.toggle('hidden', !state.replay_panels?.length);
    updateCameraTiles(S.robot.steps[S.idx]);
    drawTicks();
  }
}

// ------------------------------------------------------------------------ axis
function stageOf(node) {
  if (!node.stage) return { status: 'done' };
  return S.state.stages.find((s) => s.key === node.stage) || { status: 'pending' };
}

function minutes(s) { return s < 60 ? `${Math.round(s)} s` : `${Math.round(s / 60)} min`; }

function renderAxis() {
  const parts = [];
  NODES.forEach((node, k) => {
    const st = stageOf(node);
    if (k > 0) {
      const cls = st.status === 'done' ? 'done' : st.status === 'running' ? 'running' : '';
      parts.push(`<div class="link ${cls}"></div>`);
    }
    const firstOfStage = NODES.findIndex((n) => n.stage === node.stage) === k;
    let t = '';
    if (node.stage && firstOfStage) {
      if (st.status === 'running' && st.started) t = minutes(S.state.now - st.started);
      else if (st.seconds) t = minutes(st.seconds);
    }
    parts.push(`<div class="node ${st.status} ${node.view ? 'viewable' : ''} ${S.open === node.id ? 'open' : ''}" data-id="${node.id}" title="${node.view ? 'click to view' : ''}">
      <div class="dot"></div><div class="name">${node.label}</div><div class="t">${t}</div></div>`);
  });
  $('axis').innerHTML = parts.join('');
  const running = S.state.stages.find((s) => s.status === 'running');
  const node = running && NODES.find((n) => n.stage === running.key);
  const failed = S.state.stages.find((s) => s.status === 'failed');
  $('activity').textContent = failed ? `stage ${failed.key} failed${failed.error ? `: ${failed.error.slice(0, 160)}` : ''}`
    : running ? `${node.label}${running.activity ? ` · ${running.activity}` : ''}`
    : S.state.stages.every((s) => ['done', 'skipped'].includes(s.status)) ? 'finished' : '';
}

// ----------------------------------------------------------------------- panels
function openPanel(id) {
  S.open = id;
  S.panelKey = '';
  S.selected = null;
  $('panel').classList.toggle('hidden', !id);
  $('capture-view').classList.toggle('hidden', id !== 'capture');
  if (id === 'capture') initCapture();
  if (!id) S.playing = false;
  renderAxis();
  if (id) {
    $('panel-title').textContent = NODES.find((n) => n.id === id).title;
    renderPanel(true);
  }
}

function renderPanel(force) {
  const product = { capture: null, frames: S.state.frames, support: S.state.support, objects: S.state.objects, physics: S.state.physics }[S.open];
  const key = JSON.stringify([S.open, product, S.selected]);
  if (!force && key === S.panelKey) return;
  S.panelKey = key;
  $('panel-body').innerHTML = PANELS[S.open] ? PANELS[S.open]() : '';
}

const notYet = (stage) => `<div class="empty">Not ready yet (stage ${stage} is ${esc((S.state.stages.find((s) => s.key === stage) || {}).status || 'pending')}).</div>`;
const card = (path, caption, w = 480) => (inRun(path)
  ? `<div class="card"><img loading="lazy" src="${thumbUrl(path, w)}" data-full="${fileUrl(path)}" data-caption="${esc(path)}"><div class="cap">${caption}</div></div>`
  : '');

const PANELS = {
  capture: () => '',

  frames: () => {
    const frames = S.state.frames;
    if (!frames?.length) return notYet('2');
    return `<div class="grid">${frames.map((f) => {
      const fr = S.frameById[f.id];
      if (!fr) return '';
      const tag = f.selected ? '<span class="iou-good" title="the frame the scene was rebuilt from">selected</span>' : '';
      return `<div class="card ${f.selected ? 'selected' : ''}" title="${esc(f.note)}"><img loading="lazy" src="${thumbUrl(fr.image, 640)}" data-full="${fileUrl(fr.image)}" data-caption="${esc(f.id)}${f.note ? ` · ${esc(f.note)}` : ''}"><div class="cap"><b>${esc(f.id)}</b>${tag}</div></div>`;
    }).join('')}</div>`;
  },

  support: () => {
    const s = S.state.support;
    if (!s) return notYet('2');
    const line = `<div class="summary-line"><b>${s.extent ? s.extent.map((v) => fmt(v)).join(' × ') : '—'} m</b> · tilt <b>${fmt(s.tilt_deg, 1)}°</b>${s.rms_m == null ? '' : ` · plane RMS <b>${fmt(s.rms_m * 1000, 1)} mm</b>`}${s.description ? ` · ${esc(s.description)}` : ''}</div>`;
    return line + `<div class="grid">${s.overlays.map((p) => card(p, esc(p.split('_').pop().replace('.png', '')), 640)).join('')}</div>`;
  },

  objects: () => {
    const objs = S.state.objects;
    if (!objs?.length) return notYet('3');
    let html = `<div class="grid">${objs.map((o, k) => {
      const src = inRun(o.preview) ? thumbUrl(o.preview, 480) : inRun(o.glb) ? `/preview/${enc(o.glb)}?up=${encodeURIComponent(o.up)}&${Q}` : '';
      const iou = o.iou != null ? `<span class="${iouClass(o.iou)}" title="mean silhouette IoU over the fitted frames">IoU ${fmt(o.iou)}</span>` : '';
      const joints = o.joints ? `<span class="muted" title="articulated: parts that move">${o.joints} joint${o.joints > 1 ? 's' : ''}</span>` : '';
      return `<div class="card pick ${S.selected === k ? 'selected' : ''}" data-object="${k}">${src ? `<img src="${src}" data-object="${k}">` : ''}<div class="cap"><b>${esc(o.name)}</b>${joints}${iou}</div></div>`;
    }).join('')}</div>`;
    const o = objs[S.selected];
    if (o) {
      const model = o.model
        ? `<button data-model="${esc(o.model)}" data-name="${esc(o.name)}">3D model, joints</button>`
        : inRun(o.glb) ? `<button data-glb="${esc(o.glb)}" data-up="${esc(o.up)}" data-caption="${esc(o.name)}">3D model</button>` : '';
      html += `<div class="detail"><div class="summary-line"><b>${esc(o.name)}</b> ${model}</div>`;
      html += o.overlays.length
        ? `<div class="grid">${o.overlays.map((v) => card(v.path, `${esc(v.frame)} <span class="${iouClass(v.iou)}">IoU ${fmt(v.iou)}</span>`, 640)).join('')}</div>`
        : `<div class="empty">${S.state.method === 'fixed' ? 'the fixed method fits no single views' : 'no fit yet'}</div>`;
      html += '</div>';
    }
    return html;
  },

  physics: () => {
    const rows = S.state.physics;
    if (!rows?.length) return notYet('4');
    const span = (r) => (r ? ` (${fmt(r[0], 3)} to ${fmt(r[1], 3)})` : '');
    const dynamics = (j) => (j.type === 'prismatic' ? ['N s/m', 'N', 'N/m'] : ['N m s/rad', 'N m', 'N m/rad'])
      .map((unit, k) => [['damping', 'friction', 'stiffness'][k], unit])
      .filter(([key]) => j[key] != null)
      .map(([key, unit]) => `${key} ${fmt(j[key], 3)}${span(j.ranges?.[key])} ${unit}`)
      .concat(j.stiffness ? [`rest ${jointValue(j.type, j.rest ?? 0)}${span(j.ranges?.rest)}`] : []).join(' · ');
    const joint = (j) => `<div class="phys-item"><b>${esc(j.name)}</b> <span class="muted">${esc(j.type)}
      · ${j.limits ? j.limits.map((v) => jointValue(j.type, v)).join(' to ') : '—'} · recorded ${jointValue(j.type, j.position)}</span>
      ${dynamics(j) ? `<div class="muted">${dynamics(j)}</div>` : ''}
      ${j.why ? `<p class="why">${esc(j.why)}</p>` : ''}</div>`;
    return rows.map((r) => `<details class="phys"><summary><span class="n">${esc(r.name)}</span>
      <span class="muted">${fmt(r.mass, 3)}${span(r.ranges?.mass)} kg · friction ${fmt(r.friction)}${span(r.ranges?.friction)}${r.joints.length ? ` · ${r.joints.length} joint${r.joints.length > 1 ? 's' : ''}` : ''}</span></summary>
      <p>${esc(r.why || 'no reason written')}</p>${r.joints.map(joint).join('')}</details>`).join('');
  },
};

function onPanelClick(e) {
  const t = e.target;
  if (t.dataset.glb) { openModel(`/glb/${enc(t.dataset.glb)}?${Q}`, t.dataset.caption, t.dataset.up); return; }
  if (t.dataset.model) {
    openModel(`/scene.glb?path=${encodeURIComponent(t.dataset.model)}&object=${encodeURIComponent(t.dataset.name)}&${Q}`, t.dataset.name, 'z');
    return;
  }
  const pick = t.closest('[data-object]');
  if (pick) {
    const k = +pick.dataset.object;
    S.selected = S.selected === k ? null : k;
    renderPanel(true);
    return;
  }
  if (t.tagName === 'IMG' && t.dataset.full) openLightbox(t.dataset.full, t.dataset.caption);
}

// ---------------------------------------------------------------------- capture
async function initCapture() {
  if (S.captureReady) return;
  S.captureReady = true;
  buildCameraTiles();
  init3d();
  S.robot = await (await fetch(`/api/robot.json?${Q}`)).json();
  $('slider').max = S.robot.steps.length - 1;
  loadRobot();
  $('slider').addEventListener('input', (e) => { S.playing = false; $('play').textContent = '▶'; setIdx(+e.target.value); S.t = S.robot.times[S.idx]; });
  $('play').addEventListener('click', togglePlay);
  $('speed').addEventListener('change', (e) => { S.speed = +e.target.value; });
  $('show-replay').addEventListener('change', () => setIdx(S.idx));
  for (const id of ['show-objects', 'show-cameras', 'show-robot']) $(id).addEventListener('change', applyVisibility);
  $('scene-select').addEventListener('change', (e) => { S.sceneUserPicked = true; loadScene(e.target.value); });
  $('ticks').addEventListener('click', (e) => {
    const r = e.target.getBoundingClientRect();
    setIdx(Math.round(((e.clientX - r.left) / r.width) * (S.robot.steps.length - 1)));
  });
  window.addEventListener('resize', drawTicks);
  updateSceneSelect(S.state.scenes);
  $('replay-toggle').classList.toggle('hidden', !S.state.replay_panels?.length);
  setIdx(0);
  requestAnimationFrame(loop);
}

function frameAt(role, step) {
  const frames = S.framesByRole[role] || [];
  let best = frames[0];
  for (const f of frames) { if (f.step <= step) best = f; else break; }
  return best;
}

function setIdx(i) {
  if (!S.robot) return;
  S.idx = Math.max(0, Math.min(i, S.robot.steps.length - 1));
  $('slider').value = S.idx;
  const step = S.robot.steps[S.idx];
  const [s0, s1] = S.rec.static_steps;
  $('readout').textContent = `step ${step} · ${fmt(S.robot.times[S.idx])} s · gripper ${[].concat(S.robot.gripper[S.idx]).map((g) => fmt(g)).join(' / ')}${step >= s0 && step < s1 ? '' : ' · moving'}`;
  updateCameraTiles(step);
  poseRobot(S.idx);
  updateFrusta(step);
  drawTicks();
}

function togglePlay() {
  if (!S.robot) return;
  S.playing = !S.playing;
  $('play').textContent = S.playing ? '❚❚' : '▶';
  if (S.playing && S.idx >= S.robot.steps.length - 1) S.idx = 0;
  S.t = S.robot.times[S.idx];
}

let lastFrame = performance.now();
function loop(now) {
  const dt = (now - lastFrame) / 1000;
  lastFrame = now;
  if (S.playing && S.robot && S.open === 'capture') {
    S.t += dt * S.speed;
    const times = S.robot.times;
    if (S.t > times[times.length - 1]) S.t = 0;
    let i = times[S.idx] > S.t ? 0 : S.idx;
    while (i + 1 < times.length && times[i + 1] <= S.t) i++;
    if (i !== S.idx) setIdx(i);
  }
  if (S.open === 'capture') { controls.update(); renderer.render(scene3, camera3); }
  requestAnimationFrame(loop);
}

function drawTicks() {
  const c = $('ticks');
  if (!S.robot || !c.clientWidth) return;
  const w = c.clientWidth, h = 14, dpr = window.devicePixelRatio || 1;
  c.width = w * dpr; c.height = h * dpr;
  const g = c.getContext('2d');
  g.scale(dpr, dpr);
  const n = S.robot.steps.length - 1 || 1;
  const x = (step) => { let i = S.robot.steps.findIndex((s) => s >= step); if (i < 0) i = n; return 8 + (i / n) * (w - 16); };
  const [s0, s1] = S.rec.static_steps;
  g.fillStyle = 'rgba(63,185,122,0.18)';
  g.fillRect(x(s0), 0, Math.max(2, x(s1 - 1) - x(s0)), h);
  for (const f of S.rec.frames) {
    const chosen = S.chosen.has(f.id);
    g.fillStyle = chosen ? '#ffffff' : S.roleColor[f.camera];
    g.fillRect(x(f.step) - (chosen ? 1.5 : 0.5), chosen ? 0 : 4, chosen ? 3 : 1, chosen ? h : h - 8);
  }
  g.fillStyle = '#ff5a5a';
  g.fillRect(x(S.robot.steps[S.idx]) - 1, 0, 2, h);
}

function buildCameraTiles() {
  const box = $('cameras');
  box.innerHTML = '';
  for (const cam of S.rec.cameras) {
    const tile = document.createElement('div');
    tile.className = 'cam';
    tile.id = `cam-${cam.role}`;
    tile.innerHTML = `<img alt=""><span class="label" style="color:${S.roleColor[cam.role]}"></span>`;
    tile.querySelector('img').addEventListener('click', () => openLightbox(tile.dataset.full, tile.dataset.caption));
    box.appendChild(tile);
    for (const f of S.framesByRole[cam.role] || []) new Image().src = thumbUrl(f.image, 640);
  }
}

function updateCameraTiles(step) {
  for (const cam of S.rec.cameras) {
    const tile = $(`cam-${cam.role}`);
    const f = frameAt(cam.role, step);
    if (!tile || !f) continue;
    const panel = $('show-replay').checked ? S.replayPanels[`${cam.role}@${f.step}`] : null;
    const src = panel ? thumbUrl(panel, 1600) : thumbUrl(f.image, 640);
    const img = tile.querySelector('img');
    if (img.getAttribute('src') !== src) img.src = src;
    tile.dataset.full = fileUrl(panel || f.image);
    tile.dataset.caption = f.id;
    tile.style.gridColumn = panel ? '1 / -1' : '';
    tile.querySelector('.label').textContent = `${f.id}${S.chosen.has(f.id) ? ' · chosen' : ''}${panel ? ' · real | sim | blend | depth residual' : ''}`;
    tile.classList.toggle('chosen', S.chosen.has(f.id));
  }
}

// ------------------------------------------------------------------------- 3D
let renderer, scene3, camera3, controls, robotRoot, robotNodes = [], sceneRoot;
const frusta = {};
const gltf = new GLTFLoader();
const textures = new Map();

function init3d() {
  const el = $('view3d');
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  el.appendChild(renderer.domElement);
  scene3 = new THREE.Scene();
  scene3.background = new THREE.Color(0x0a0c10);
  camera3 = new THREE.PerspectiveCamera(45, 4 / 3, 0.01, 50);
  camera3.position.set(1.7, -1.3, 1.1);
  controls = new OrbitControls(camera3, renderer.domElement);
  controls.target.set(0.45, 0, 0.1);
  scene3.add(new THREE.HemisphereLight(0xffffff, 0x404040, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 1.6);
  sun.position.set(2, -1.5, 3);
  scene3.add(sun);
  const grid = new THREE.GridHelper(2, 20, 0x39424f, 0x222831);
  grid.rotation.x = Math.PI / 2;
  scene3.add(grid, new THREE.AxesHelper(0.15));
  robotRoot = new THREE.Group();
  sceneRoot = new THREE.Group();
  scene3.add(robotRoot, sceneRoot);
  for (const cam of S.rec.cameras) { frusta[cam.role] = makeFrustum(cam); scene3.add(frusta[cam.role]); }
  new ResizeObserver(() => {
    const w = el.clientWidth, h = el.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h);
    camera3.aspect = w / h;
    camera3.updateProjectionMatrix();
    drawTicks();
  }).observe(el);
}

function loadRobot() {
  gltf.load(`/robot.glb?${Q}`, (g) => {
    robotRoot.add(g.scene);
    robotNodes = S.robot.bodies.map((_, k) => g.scene.getObjectByName(`b${k}`));
    g.scene.traverse((o) => { if (o.isMesh) { o.material.metalness = 0.1; o.material.roughness = 0.6; } });
    poseRobot(S.idx);
  });
}

function poseRobot(i) {
  if (!S.robot || !robotNodes.length) return;
  S.robot.poses[i].forEach((p, k) => {
    const node = robotNodes[k];
    if (!node) return;
    node.position.set(p[0], p[1], p[2]);
    node.quaternion.set(p[3], p[4], p[5], p[6]);
  });
}

function makeFrustum(cam) {
  const d = cam.static ? 0.15 : 0.08;
  const K = cam.K;
  const corner = (u, v) => new THREE.Vector3(((u - K[0][2]) / K[0][0]) * d, ((v - K[1][2]) / K[1][1]) * d, d);
  const c = [corner(0, 0), corner(cam.width, 0), corner(cam.width, cam.height), corner(0, cam.height)];
  const o = new THREE.Vector3();
  const group = new THREE.Group();
  group.matrixAutoUpdate = false;
  group.add(new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints([o, c[0], o, c[1], o, c[2], o, c[3], c[0], c[1], c[1], c[2], c[2], c[3], c[3], c[0]]),
    new THREE.LineBasicMaterial({ color: S.roleColor[cam.role] }),
  ));
  const quad = new THREE.BufferGeometry().setFromPoints(c);
  quad.setAttribute('uv', new THREE.Float32BufferAttribute([0, 1, 1, 1, 1, 0, 0, 0], 2));
  quad.setIndex([0, 1, 2, 0, 2, 3]);
  const image = new THREE.Mesh(quad, new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, transparent: true, opacity: 0.9 }));
  image.name = 'image';
  group.add(image);
  return group;
}

function updateFrusta(step) {
  for (const cam of S.rec.cameras) {
    const f = frameAt(cam.role, step);
    const group = frusta[cam.role];
    if (!f || !group) continue;
    group.matrix.copy(new THREE.Matrix4().set(...f.T_base_cam.flat()));
    group.matrixWorldNeedsUpdate = true;
    const url = thumbUrl(f.image, 320);
    let tex = textures.get(url);
    if (!tex) {
      tex = new THREE.TextureLoader().load(url);
      tex.colorSpace = THREE.SRGBColorSpace;
      textures.set(url, tex);
    }
    const mesh = group.getObjectByName('image');
    mesh.material.map = tex;
    mesh.material.needsUpdate = true;
  }
  applyVisibility();
}

function applyVisibility() {
  robotRoot.visible = $('show-robot').checked;
  sceneRoot.visible = $('show-objects').checked;
  for (const g of Object.values(frusta)) g.visible = $('show-cameras').checked;
}

// Converted OBJ materials can come out fully metallic, which renders black without an
// environment map; an open surface is seen from both sides.
function matte(material) {
  for (const m of [].concat(material)) {
    if ('metalness' in m) { m.metalness = 0; m.roughness = 0.8; }
    m.side = THREE.DoubleSide;
  }
}

function loadScene(path) {
  S.scenePath = path;
  $('joints').innerHTML = '';
  while (sceneRoot.children.length) sceneRoot.remove(sceneRoot.children[0]);
  if (!path) return;
  gltf.load(`/scene.glb?path=${encodeURIComponent(path)}&${Q}`, (g) => {
    if (S.scenePath !== path) return;
    g.scene.traverse((o) => {
      if (!o.isMesh) return;
      matte(o.material);
      if (o.name.startsWith('support')) { o.material.transparent = true; o.material.opacity = 0.45; o.material.depthWrite = false; }
    });
    sceneRoot.add(g.scene);
    articulate(g.scene, $('joints'));
  });
}

// ------------------------------------------------------------------ articulation
// The joints of a scene's articulated objects come in its GLB's extras (see
// viewer/scenes.py): each with its axis in the base frame as recorded, its limits and
// recorded position, the nodes it moves and a node of its parent link. A slider per
// joint turns or slides those nodes; parents go first, so a joint's axis follows its
// parent's motion. The axes are drawn as lines.
function articulate(root, box) {
  box.innerHTML = '';
  const joints = root.userData.joints || [];
  if (!joints.length) return;
  const values = joints.map((j) => j.position);
  const axes = joints.map(() => {
    const line = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xff5ad2, depthTest: false }));
    line.renderOrder = 1; // drawn over the parts it lies on
    root.add(line);
    return line;
  });
  const apply = () => {
    const motion = {}; // node -> its motion from the recorded pose
    joints.forEach((j, k) => {
      const parent = (j.parent_node && motion[j.parent_node]) || new THREE.Matrix4();
      const p = new THREE.Vector3(...j.origin).applyMatrix4(parent);
      const a = new THREE.Vector3(...j.axis).transformDirection(parent);
      const d = values[k] - j.position;
      const m = j.type === 'prismatic'
        ? new THREE.Matrix4().makeTranslation(a.clone().multiplyScalar(d))
        : new THREE.Matrix4().makeTranslation(p).multiply(new THREE.Matrix4().makeRotationAxis(a, d)).multiply(new THREE.Matrix4().makeTranslation(p.clone().negate()));
      for (const n of j.nodes) motion[n] = m.clone().multiply(motion[n] || new THREE.Matrix4());
      axes[k].geometry.setFromPoints([p.clone().addScaledVector(a, -0.12), p.clone().addScaledVector(a, 0.12)]);
    });
    for (const [name, m] of Object.entries(motion)) {
      const node = root.getObjectByName(name);
      if (!node) continue;
      node.matrixAutoUpdate = false;
      node.matrix.copy(m);
      node.matrixWorldNeedsUpdate = true;
    }
  };
  box.innerHTML = joints.map((j, k) => {
    const [lo, hi] = j.limits;
    return `<label title="${esc(j.type)} joint; limits ${jointValue(j.type, lo)} to ${jointValue(j.type, hi)}">${esc(j.object)} · ${esc(j.name)}
      <input type="range" data-k="${k}" min="${lo}" max="${hi}" step="${(hi - lo) / 200}" value="${j.position}">
      <span class="v" id="${box.id}-v${k}">${jointValue(j.type, j.position)}</span></label>`;
  }).join('');
  box.querySelectorAll('input').forEach((input) => input.addEventListener('input', () => {
    const k = +input.dataset.k;
    values[k] = +input.value;
    $(`${box.id}-v${k}`).textContent = jointValue(joints[k].type, values[k]);
    apply();
  }));
  apply();
}

function updateSceneSelect(scenes) {
  const sel = $('scene-select');
  const key = JSON.stringify(scenes.map((s) => [s.path, s.mtime]));
  if (sel.dataset.key === key) return;
  sel.dataset.key = key;
  sel.innerHTML = scenes.length ? scenes.map((s) => `<option value="${esc(s.path)}">${esc(s.label)}</option>`).join('') : '<option value="">(no scene yet)</option>';
  const want = S.sceneUserPicked && scenes.some((s) => s.path === S.scenePath) ? S.scenePath : scenes[0]?.path || '';
  sel.value = want;
  loadScene(want);
}

// --------------------------------------------------------------------- lightbox
let lbRenderer = null;
let lbResize = null;
function openLightbox(src, caption) {
  if (!src) return;
  $('lb-img').classList.remove('hidden');
  $('lb-model').classList.add('hidden');
  $('lb-joints').innerHTML = '';
  $('lb-img').src = src;
  $('lb-caption').textContent = caption || '';
  $('lightbox').classList.remove('hidden');
}

// A model from ``url`` (a GLB) turned ``up``-axis up; an articulated object's joints
// get sliders below it.
function openModel(url, caption, up) {
  $('lb-img').classList.add('hidden');
  const box = $('lb-model');
  box.classList.remove('hidden');
  box.innerHTML = '';
  $('lb-joints').innerHTML = '';
  $('lb-caption').textContent = `${caption} (drag to turn)`;
  $('lightbox').classList.remove('hidden');
  lbRenderer = new THREE.WebGLRenderer({ antialias: true });
  box.appendChild(lbRenderer.domElement);
  const sc = new THREE.Scene();
  sc.background = new THREE.Color(0x15181d);
  sc.add(new THREE.HemisphereLight(0xffffff, 0x444444, 2.0));
  const cam = new THREE.PerspectiveCamera(40, 1, 0.001, 100);
  const ctl = new OrbitControls(cam, lbRenderer.domElement);
  // The box's size changes as the joint sliders appear below it.
  lbResize = new ResizeObserver(() => {
    const w = box.clientWidth, h = box.clientHeight;
    if (!w || !h || !lbRenderer) return;
    lbRenderer.setSize(w, h, false);
    cam.aspect = w / h;
    cam.updateProjectionMatrix();
  });
  lbResize.observe(box);
  gltf.load(url, (g) => {
    const obj = g.scene;
    obj.traverse((o) => { if (o.isMesh) matte(o.material); });
    obj.rotation.set(...(UP_EULER[up] || UP_EULER.z));
    sc.add(obj);
    articulate(obj, $('lb-joints'));
    const bounds = new THREE.Box3().setFromObject(obj);
    const size = bounds.getSize(new THREE.Vector3()).length();
    const centre = bounds.getCenter(new THREE.Vector3());
    ctl.target.copy(centre);
    cam.position.copy(centre).add(new THREE.Vector3(size, -size, size * 0.7));
  });
  const r = lbRenderer;
  (function spin() {
    if (r !== lbRenderer) return;
    ctl.update();
    r.render(sc, cam);
    requestAnimationFrame(spin);
  })();
}

function closeLightbox() {
  $('lightbox').classList.add('hidden');
  if (lbResize) { lbResize.disconnect(); lbResize = null; }
  if (lbRenderer) { lbRenderer.dispose(); lbRenderer = null; }
}

init();
