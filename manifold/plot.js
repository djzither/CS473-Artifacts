/* ============================================================================
   Manifold Learning Playground — drawing primitives.
   Viewer3D: a small canvas 3-D viewer (drag to rotate, wheel / pinch to zoom,
             painter's-algorithm depth sort).
   Scatter2D: an equal-aspect 2-D scatter with animated transitions and hover.
   ========================================================================= */

(function (global) {
  'use strict';

  /* ---------------------------------------------------------------- colors */
  var VIR = ['#440154', '#482878', '#3e4989', '#31688e', '#26828e', '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725']
    .map(function (h) { return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; });
  function clamp(x, a, b) { return x < a ? a : x > b ? b : x; }
  function rgbStr(c, a) { return a == null ? 'rgb(' + (c[0] | 0) + ',' + (c[1] | 0) + ',' + (c[2] | 0) + ')' : 'rgba(' + (c[0] | 0) + ',' + (c[1] | 0) + ',' + (c[2] | 0) + ',' + a + ')'; }
  /* Viridis, stopped at 90% so the palest yellow never sits on a white page. */
  function viridisRGB(t) {
    var x = clamp(t, 0, 1) * 0.9 * (VIR.length - 1), i = Math.min(Math.floor(x), VIR.length - 2), f = x - i, a = VIR[i], b = VIR[i + 1];
    return [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])];
  }
  /* A cyclic map for the circle, so angle 0 and 2π get the same color. */
  function sinebowRGB(t) {
    t = 0.5 - t;
    var r = Math.sin(Math.PI * t), g = Math.sin(Math.PI * (t + 1 / 3)), b = Math.sin(Math.PI * (t + 2 / 3));
    return [215 * r * r + 20, 200 * g * g + 20, 215 * b * b + 20];
  }
  function viridis(t, a) { return rgbStr(viridisRGB(t), a); }
  function sinebow(t, a) { return rgbStr(sinebowRGB(t), a); }

  function setupCanvas(canvas) {
    var r = canvas.getBoundingClientRect(), dpr = Math.min(global.devicePixelRatio || 1, 2);
    var w = Math.max(1, Math.round(r.width)), h = Math.max(1, Math.round(r.height));
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
    var ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: w, h: h };
  }

  /* ============================================================== Viewer3D */
  function Viewer3D(canvas, opt) {
    opt = opt || {};
    var self = {
      yaw: opt.yaw != null ? opt.yaw : -0.7, pitch: opt.pitch != null ? opt.pitch : 0.35,
      zoom: 1, center: [0, 0, 0], radius: 1, scene: null, onClick: null, onHover: null, onRender: null
    };
    var W = 1, H = 1, raf = 0, spin = false, tween = null, lastFrame = 0, home = null;

    function project(p) {
      var x = p[0] - self.center[0], y = p[1] - self.center[1], z = p[2] - self.center[2];
      var cy = Math.cos(self.yaw), sy = Math.sin(self.yaw);
      var x1 = cy * x + sy * z, z1 = -sy * x + cy * z;
      var cp = Math.cos(self.pitch), sp = Math.sin(self.pitch);
      var y2 = cp * y - sp * z1, z2 = sp * y + cp * z1;
      var D = 4.5 * self.radius / Math.max(1, Math.sqrt(self.zoom)), per = D / Math.max(D - z2, 0.15 * D);
      var s = self.zoom * Math.min(W, H) / (2.25 * self.radius);
      return [W / 2 + s * x1 * per, H / 2 - s * y2 * per, z2, per];
    }
    self.project = project;
    self.size = function () { return [W, H]; };

    self.setBounds = function (center, radius) {
      self.center = center.slice(); self.radius = radius;
      home = { center: center.slice(), zoom: 1 };
      self.render();
    };
    self.boundsOf = function (pts) {
      var lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      pts.forEach(function (p) { for (var a = 0; a < 3; a++) { lo[a] = Math.min(lo[a], p[a]); hi[a] = Math.max(hi[a], p[a]); } });
      var c = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2], r = 0;
      pts.forEach(function (p) { r = Math.max(r, Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2])); });
      self.setBounds(c, r || 1);
    };
    self.render = function () { if (!raf) raf = requestAnimationFrame(frame); };
    self.setSpin = function (on) { spin = on; lastFrame = 0; self.render(); };
    self.spinning = function () { return spin; };
    self.animateTo = function (center, zoom, ms) {
      tween = { c0: self.center.slice(), z0: self.zoom, c1: center.slice(), z1: zoom, t0: performance.now(), ms: ms || 700 };
      self.render();
    };
    self.home = function () { if (home) self.animateTo(home.center, home.zoom); };

    function frame(t) {
      raf = 0;
      if (spin) { var dt = lastFrame ? Math.min(t - lastFrame, 50) : 16; self.yaw += dt * 0.00045; lastFrame = t; }
      if (tween) {
        var a = clamp((t - tween.t0) / tween.ms, 0, 1), e = a < 0.5 ? 2 * a * a : 1 - Math.pow(-2 * a + 2, 2) / 2;
        for (var k = 0; k < 3; k++) self.center[k] = tween.c0[k] + e * (tween.c1[k] - tween.c0[k]);
        self.zoom = Math.exp(Math.log(tween.z0) + e * (Math.log(tween.z1) - Math.log(tween.z0)));
        if (a >= 1) tween = null;
      }
      draw();
      if (spin || tween) self.render();
    }

    function draw() {
      var c = setupCanvas(canvas), ctx = c.ctx; W = c.w; H = c.h;
      ctx.clearRect(0, 0, W, H);
      var prims = [];
      var api = {
        project: project,
        quad: function (pts, fill, stroke) {
          var q = pts.map(project), z = 0; q.forEach(function (p) { z += p[2]; });
          prims.push({ k: 0, z: z / q.length, q: q, fill: fill, stroke: stroke });
        },
        line: function (a, b, color, width, o) {
          var pa = project(a), pb = project(b);
          prims.push({ k: 1, z: (o && o.top ? 1e9 : 0) + (pa[2] + pb[2]) / 2, a: pa, b: pb, color: color, width: width || 1, dash: o && o.dash });
        },
        polyline: function (pts, color, width, o) { for (var i = 0; i + 1 < pts.length; i++) api.line(pts[i], pts[i + 1], color, width, o); },
        point: function (p, r, fill, o) {
          var q = project(p);
          prims.push({ k: 2, z: (o && o.top ? 2e9 : 0) + q[2], p: q, r: r, fill: fill, stroke: o && o.stroke, lw: (o && o.lw) || 1.5 });
        },
        label: function (p, text, color) { var q = project(p); prims.push({ k: 3, z: 3e9, p: q, text: text, color: color }); }
      };
      if (self.scene) self.scene(api);
      prims.sort(function (a, b) { return a.z - b.z; });
      ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      for (var i = 0; i < prims.length; i++) {
        var m = prims[i];
        if (m.k === 0) {
          ctx.beginPath(); ctx.moveTo(m.q[0][0], m.q[0][1]);
          for (var j = 1; j < m.q.length; j++) ctx.lineTo(m.q[j][0], m.q[j][1]);
          ctx.closePath(); ctx.fillStyle = m.fill; ctx.fill();
          ctx.strokeStyle = m.stroke || m.fill; ctx.lineWidth = m.stroke ? 0.6 : 0.8; ctx.stroke();
        } else if (m.k === 1) {
          ctx.beginPath(); ctx.moveTo(m.a[0], m.a[1]); ctx.lineTo(m.b[0], m.b[1]);
          ctx.strokeStyle = m.color; ctx.lineWidth = m.width; ctx.setLineDash(m.dash || []); ctx.stroke(); ctx.setLineDash([]);
        } else if (m.k === 2) {
          ctx.beginPath(); ctx.arc(m.p[0], m.p[1], m.r, 0, 2 * Math.PI);
          ctx.fillStyle = m.fill; ctx.fill();
          if (m.stroke) { ctx.strokeStyle = m.stroke; ctx.lineWidth = m.lw; ctx.stroke(); }
        } else {
          ctx.font = '600 12px "IBM Plex Mono", monospace'; ctx.lineWidth = 4; ctx.strokeStyle = 'rgba(255,255,255,0.92)';
          ctx.strokeText(m.text, m.p[0] + 8, m.p[1] - 8); ctx.fillStyle = m.color; ctx.fillText(m.text, m.p[0] + 8, m.p[1] - 8);
        }
      }
      if (self.onRender) self.onRender(ctx, W, H);
    }

    /* ---- interaction ---- */
    var ptrs = new Map(), down = null, moved = 0, pinch = null;
    function local(e) { var r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
    canvas.addEventListener('pointerdown', function (e) {
      canvas.setPointerCapture(e.pointerId);
      ptrs.set(e.pointerId, [e.clientX, e.clientY]);
      if (ptrs.size === 1) { down = { x: e.clientX, y: e.clientY, yaw: self.yaw, pitch: self.pitch }; moved = 0; }
      else if (ptrs.size === 2) {
        var v = Array.from(ptrs.values()); pinch = { d: Math.hypot(v[0][0] - v[1][0], v[0][1] - v[1][1]), z: self.zoom }; moved = 99;
      }
    });
    canvas.addEventListener('pointermove', function (e) {
      if (ptrs.has(e.pointerId)) {
        ptrs.set(e.pointerId, [e.clientX, e.clientY]);
        if (ptrs.size === 2 && pinch) {
          var v = Array.from(ptrs.values()), d = Math.hypot(v[0][0] - v[1][0], v[0][1] - v[1][1]);
          self.zoom = clamp(pinch.z * d / Math.max(pinch.d, 1), 0.4, 14); tween = null; self.render();
        } else if (down) {
          var dx = e.clientX - down.x, dy = e.clientY - down.y;
          moved = Math.max(moved, Math.abs(dx) + Math.abs(dy));
          self.yaw = down.yaw + dx * 0.01; self.pitch = clamp(down.pitch + dy * 0.01, -1.5, 1.5); self.render();
        }
      } else if (self.onHover) { var p = local(e); self.onHover(p[0], p[1]); }
    });
    function up(e) {
      if (ptrs.size === 1 && moved < 5 && self.onClick) { var p = local(e); self.onClick(p[0], p[1]); }
      ptrs.delete(e.pointerId);
      if (ptrs.size < 2) pinch = null;
      if (!ptrs.size) down = null;
    }
    canvas.addEventListener('pointerup', up);
    canvas.addEventListener('pointercancel', function (e) { ptrs.delete(e.pointerId); down = null; pinch = null; });
    canvas.addEventListener('pointerleave', function () { if (!ptrs.size && self.onHover) self.onHover(-1, -1); });
    canvas.addEventListener('wheel', function (e) {
      e.preventDefault(); tween = null;
      self.zoom = clamp(self.zoom * Math.exp(-e.deltaY * 0.0015), 0.4, 14); self.render();
    }, { passive: false });
    canvas.addEventListener('keydown', function (e) {
      var used = true;
      if (e.key === 'ArrowLeft') self.yaw -= 0.1; else if (e.key === 'ArrowRight') self.yaw += 0.1;
      else if (e.key === 'ArrowUp') self.pitch = clamp(self.pitch - 0.1, -1.5, 1.5); else if (e.key === 'ArrowDown') self.pitch = clamp(self.pitch + 0.1, -1.5, 1.5);
      else if (e.key === '+' || e.key === '=') self.zoom = clamp(self.zoom * 1.15, 0.4, 14); else if (e.key === '-') self.zoom = clamp(self.zoom / 1.15, 0.4, 14);
      else used = false;
      if (used) { e.preventDefault(); self.render(); }
    });
    if (global.ResizeObserver) new ResizeObserver(function () { self.render(); }).observe(canvas);
    return self;
  }

  /* Nearest projected point to a screen position, front-most among near ties. */
  function pick(viewer, pts, x, y, tol) {
    var t2 = (tol || 12) * (tol || 12), best = -1, bz = -Infinity;
    for (var i = 0; i < pts.length; i++) {
      var q = viewer.project(pts[i]), d = (q[0] - x) * (q[0] - x) + (q[1] - y) * (q[1] - y);
      if (d < t2 && q[2] > bz) { best = i; bz = q[2]; }
    }
    return best;
  }

  /* ============================================================= Scatter2D */
  function Scatter2D(canvas, opt) {
    opt = opt || {};
    var self = { colors: null, r: opt.r || 3, hi: -1, nbrs: null, overlay: null, onHover: null, onClick: null, bounds: opt.bounds || null, empty: '' };
    var cur = null, from = null, to = null, t0 = 0, raf = 0, W = 1, H = 1, view = { s: 1, cx: 0, cy: 0 };

    self.set = function (Y, colors, animate) {
      var n = Y ? Y.length : 0, next = new Float64Array(2 * n);
      for (var i = 0; i < n; i++) { next[2 * i] = Y[i][0]; next[2 * i + 1] = Y[i][1]; }
      if (colors) self.colors = colors;
      if (animate && cur && cur.length === next.length && !matchMedia('(prefers-reduced-motion: reduce)').matches) { from = cur.slice(); to = next; t0 = performance.now(); }
      else { cur = next; from = to = null; }
      self.render();
    };
    self.clear = function (msg) { cur = null; from = to = null; self.empty = msg || ''; self.render(); };
    self.render = function () { if (!raf) raf = requestAnimationFrame(frame); };
    self.toScreen = function (x, y) { return [W / 2 + view.s * (x - view.cx), H / 2 - view.s * (y - view.cy)]; };

    function fit() {
      var b = self.bounds;
      if (!b) {
        var x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
        for (var i = 0; i < cur.length; i += 2) { x0 = Math.min(x0, cur[i]); x1 = Math.max(x1, cur[i]); y0 = Math.min(y0, cur[i + 1]); y1 = Math.max(y1, cur[i + 1]); }
        b = [x0, x1, y0, y1];
      }
      var pad = opt.pad != null ? opt.pad : 12, dx = Math.max(b[1] - b[0], 1e-9), dy = Math.max(b[3] - b[2], 1e-9);
      view.s = Math.min((W - 2 * pad) / dx, (H - 2 * pad) / dy);
      view.cx = (b[0] + b[1]) / 2; view.cy = (b[2] + b[3]) / 2;
    }

    function frame(t) {
      raf = 0;
      if (from) {
        var a = clamp((t - t0) / (opt.ms || 650), 0, 1), e = a < 0.5 ? 4 * a * a * a : 1 - Math.pow(-2 * a + 2, 3) / 2;
        cur = new Float64Array(to.length);
        for (var i = 0; i < to.length; i++) cur[i] = from[i] + e * (to[i] - from[i]);
        if (a >= 1) { cur = to; from = to = null; } else self.render();
      }
      var c = setupCanvas(canvas), ctx = c.ctx; W = c.w; H = c.h;
      ctx.clearRect(0, 0, W, H);
      if (!cur) {
        if (self.empty) { ctx.fillStyle = '#7b879c'; ctx.font = '500 13px "IBM Plex Sans", sans-serif'; ctx.textAlign = 'center'; ctx.fillText(self.empty, W / 2, H / 2); ctx.textAlign = 'start'; }
        return;
      }
      fit();
      if (opt.underlay) opt.underlay(ctx, self.toScreen, view.s, W, H);
      var n = cur.length / 2, r = self.r, fade = self.hi >= 0 && opt.fadeOnHover !== false;
      for (var i2 = 0; i2 < n; i2++) {
        var p = self.toScreen(cur[2 * i2], cur[2 * i2 + 1]);
        ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 2 * Math.PI);
        ctx.fillStyle = self.colors ? self.colors[i2] : '#46536b';
        ctx.globalAlpha = fade ? 0.35 : 0.92; ctx.fill();
      }
      ctx.globalAlpha = 1;
      if (self.hi >= 0 && self.hi < n) {
        if (self.nbrs) self.nbrs.forEach(function (j) {
          var q = self.toScreen(cur[2 * j], cur[2 * j + 1]);
          ctx.beginPath(); ctx.arc(q[0], q[1], r + 2.5, 0, 2 * Math.PI); ctx.fillStyle = self.colors ? self.colors[j] : '#000'; ctx.fill();
          ctx.strokeStyle = '#14213d'; ctx.lineWidth = 1.4; ctx.stroke();
        });
        var h = self.toScreen(cur[2 * self.hi], cur[2 * self.hi + 1]);
        ctx.beginPath(); ctx.arc(h[0], h[1], r + 5, 0, 2 * Math.PI); ctx.fillStyle = '#fff'; ctx.fill();
        ctx.strokeStyle = '#14213d'; ctx.lineWidth = 2.5; ctx.stroke();
        ctx.beginPath(); ctx.arc(h[0], h[1], r + 1.5, 0, 2 * Math.PI); ctx.fillStyle = self.colors ? self.colors[self.hi] : '#000'; ctx.fill();
      }
      if (self.overlay) self.overlay(ctx, self.toScreen, view.s, W, H);
    }

    self.nearest = function (x, y, tol) {
      if (!cur) return -1;
      var best = -1, bd = (tol || 10) * (tol || 10);
      for (var i = 0; i < cur.length / 2; i++) {
        var p = self.toScreen(cur[2 * i], cur[2 * i + 1]), d = (p[0] - x) * (p[0] - x) + (p[1] - y) * (p[1] - y);
        if (d < bd) { bd = d; best = i; }
      }
      return best;
    };
    function local(e) { var r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
    canvas.addEventListener('pointermove', function (e) { if (self.onHover) { var p = local(e); self.onHover(self.nearest(p[0], p[1]), p); } });
    canvas.addEventListener('pointerleave', function () { if (self.onHover) self.onHover(-1, null); });
    canvas.addEventListener('click', function (e) { if (self.onClick) { var p = local(e); self.onClick(self.nearest(p[0], p[1], 16), p, self.fromScreen(p[0], p[1])); } });
    self.fromScreen = function (x, y) { return [view.cx + (x - W / 2) / view.s, view.cy - (y - H / 2) / view.s]; };
    if (global.ResizeObserver) new ResizeObserver(function () { self.render(); }).observe(canvas);
    return self;
  }

  /* Tiny bar chart for eigenvalue spectra and similar. */
  function bars(canvas, vals, o) {
    o = o || {};
    var c = setupCanvas(canvas), ctx = c.ctx, W = c.w, H = c.h, n = vals.length, pad = 18, top = 6;
    ctx.clearRect(0, 0, W, H);
    if (!n) return;
    var mx = o.max || Math.max.apply(null, vals.map(Math.abs)) || 1, bw = (W - 8) / n;
    for (var i = 0; i < n; i++) {
      var h = Math.max(0, Math.abs(vals[i]) / mx) * (H - pad - top), x = 4 + i * bw;
      ctx.fillStyle = o.color ? o.color(i) : '#1d5fd6';
      ctx.fillRect(x + bw * 0.15, H - pad - h, bw * 0.7, Math.max(h, 1));
      ctx.fillStyle = '#7b879c'; ctx.font = '400 10.5px "IBM Plex Mono", monospace'; ctx.textAlign = 'center';
      ctx.fillText(o.labels ? o.labels[i] : String(i + 1), x + bw / 2, H - 5);
    }
    ctx.textAlign = 'start';
    ctx.strokeStyle = '#c3cede'; ctx.beginPath(); ctx.moveTo(2, H - pad + 0.5); ctx.lineTo(W - 2, H - pad + 0.5); ctx.stroke();
  }

  global.Plot = { Viewer3D: Viewer3D, Scatter2D: Scatter2D, pick: pick, bars: bars, setupCanvas: setupCanvas,
    viridis: viridis, sinebow: sinebow, viridisRGB: viridisRGB, sinebowRGB: sinebowRGB, rgbStr: rgbStr, clamp: clamp };
})(window);
