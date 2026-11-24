/* Module 7 — Exam practice: randomized questions, hints, solutions, score, weak topics. */
(function () {
  'use strict';
  var P = window.PCA, Q = window.PCAPractice, U = window.UI, $ = U.$;
  var KEY = 'pca-lab-practice-v1';

  function load() {
    try { var s = JSON.parse(localStorage.getItem(KEY)); if (s && s.topics) return s; } catch (e) { /* storage unavailable */ }
    return { total: 0, correct: 0, streak: 0, best: 0, n: 0, off: {}, topics: {} };
  }
  function save(s) { try { localStorage.setItem(KEY, JSON.stringify(s)); } catch (e) { /* ignore */ } }

  UI.module('m7', function () {
    var st = load(), q = null, graded = false, r = P.rng((Date.now() % 1e9) >>> 0);
    var name = {}; Q.TOPICS.forEach(function (t) { name[t.key] = t.name; });

    function acc(k) { var t = st.topics[k]; return t && t.att ? t.cor / t.att : null; }
    function pickTopic() {
      var keys = Q.TOPICS.map(function (t) { return t.key; }).filter(function (k) { return !st.off[k]; });
      if (!keys.length) keys = Q.TOPICS.map(function (t) { return t.key; });
      var focus = $('#m7focus').checked;
      var w = keys.map(function (k) { if (!focus) return 1; var a = acc(k); return a === null ? 1.5 : 0.3 + 4 * (1 - a); });
      if (q && keys.length > 1) w = w.map(function (x, i) { return keys[i] === q.topic ? x * 0.25 : x; });
      var tot = w.reduce(function (s, x) { return s + x; }, 0), u = r() * tot;
      for (var i = 0; i < keys.length; i++) { u -= w[i]; if (u <= 0) return keys[i]; }
      return keys[keys.length - 1];
    }

    function next() {
      q = Q.make(pickTopic(), r); graded = false; st.n = (st.n || 0) + 1; save(st);
      $('#m7topic').textContent = name[q.topic];
      $('#m7count').textContent = 'question ' + st.n;
      U.html($('#m7prompt'), q.prompt);
      var a = $('#m7answer');
      if (q.kind === 'mc') {
        var h = '<div class="opts" role="radiogroup" aria-label="Answer choices">';
        q.options.forEach(function (o, i) { h += '<label class="opt"><input type="radio" name="m7opt" value="' + i + '"> <span>' + o + '</span></label>'; });
        U.html(a, h + '</div>');
      } else {
        U.html(a, '<div class="ansrow"><label for="m7in" class="muted">Your answer</label><input id="m7in" type="text" autocomplete="off" placeholder="' + (q.kind === 'dim' ? 'e.g. 5 x 3' : q.percentOK ? 'e.g. 0.82 or 82%' : 'number') + '"></div>');
        $('#m7in').addEventListener('keydown', function (ev) { if (ev.key === 'Enter') { ev.preventDefault(); if (!graded) check(); else next(); } });
        setTimeout(function () { var i = $('#m7in'); if (i) i.focus({ preventScroll: true }); }, 0);
      }
      $('#m7hinttext').hidden = true; U.html($('#m7hinttext'), '<b>Hint.</b> ' + q.hint);
      $('#m7fb').textContent = ''; $('#m7fb').className = 'feedback';
      $('#m7sol').hidden = true;
      U.html($('#m7sol'), '<h4>Step-by-step solution</h4><ol>' + q.steps.map(function (s) { return '<li>' + s + '</li>'; }).join('') + '</ol>');
      $('#m7check').disabled = false;
    }

    function answerText() {
      if (q.kind === 'mc') return q.options[q.correct];
      if (q.kind === 'dim') return q.answer[0] + ' × ' + q.answer[1];
      var v = q.answer; return (Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : String(+v.toFixed(4))) + (q.percentOK ? ' (= ' + (100 * v).toFixed(2) + '%)' : '');
    }
    function record(ok) {
      if (graded) return; graded = true;
      var t = st.topics[q.topic] || (st.topics[q.topic] = { att: 0, cor: 0 });
      t.att++; st.total++;
      if (ok) { t.cor++; st.correct++; st.streak++; st.best = Math.max(st.best, st.streak); } else st.streak = 0;
      save(st); score();
    }
    function check() {
      if (graded) return;
      var ok;
      if (q.kind === 'mc') {
        var sel = document.querySelector('input[name=m7opt]:checked');
        if (!sel) { $('#m7fb').className = 'feedback no'; $('#m7fb').textContent = 'Pick an option first.'; return; }
        ok = +sel.value === q.correct;
        U.$$('#m7answer .opt').forEach(function (o, i) { o.classList.toggle('is-right', i === q.correct); o.classList.toggle('is-wrong', i === +sel.value && !ok); });
      } else {
        var txt = $('#m7in').value;
        if (!txt.trim()) { $('#m7fb').className = 'feedback no'; $('#m7fb').textContent = 'Type an answer first (or use Show solution).'; return; }
        ok = Q.grade(q, txt);
      }
      record(ok);
      $('#m7fb').className = 'feedback ' + (ok ? 'ok' : 'no');
      $('#m7fb').textContent = ok ? 'Correct. ' + (q.kind !== 'mc' ? 'Answer: ' + answerText() + '.' : '') : 'Not quite. The answer is ' + answerText() + '.';
      $('#m7sol').hidden = false; $('#m7check').disabled = true;
    }

    function score() {
      var pct = st.total ? Math.round(100 * st.correct / st.total) + '%' : '—';
      U.html($('#m7score'), U.stat('correct', st.correct + ' / ' + st.total, 'ok') + U.stat('accuracy', pct) + U.stat('streak (best)', st.streak + ' (' + st.best + ')'));
      var weak = Q.TOPICS.filter(function (t) { var a = acc(t.key); return a !== null && a < 0.7; })
        .sort(function (a, b) { return acc(a.key) - acc(b.key); });
      U.html($('#m7weak'), weak.length
        ? '<div class="weak"><b>Topics to review</b> (below 70%):<ul>' + weak.map(function (t) { var s = st.topics[t.key]; return '<li>' + t.name + ': ' + s.cor + '/' + s.att + '</li>'; }).join('') + '</ul></div>'
        : '<p class="note">' + (st.total ? 'No weak topics yet: every topic you have tried is at 70% or better.' : 'Your weakest topics will appear here as you answer.') + '</p>');
      var h = '';
      Q.TOPICS.forEach(function (t) {
        var s = st.topics[t.key], a = acc(t.key);
        h += '<label class="trow"><input type="checkbox" data-k="' + t.key + '"' + (st.off[t.key] ? '' : ' checked') + '><span>' + t.name + '</span><span class="tstat">' + (s ? s.cor + '/' + s.att : '–') + '</span>' +
          '<span class="tbar"><i style="width:' + (a === null ? 0 : Math.round(100 * a)) + '%"></i></span></label>';
      });
      $('#m7topics').innerHTML = h;
    }

    $('#m7topics').addEventListener('change', function (ev) { var k = ev.target.dataset.k; if (k) { if (ev.target.checked) delete st.off[k]; else st.off[k] = 1; save(st); } });
    $('#m7check').addEventListener('click', check);
    $('#m7hint').addEventListener('click', function () { $('#m7hinttext').hidden = false; });
    $('#m7reveal').addEventListener('click', function () {
      if (!graded) { record(false); $('#m7fb').className = 'feedback no'; $('#m7fb').textContent = 'Revealed (counted as a miss). The answer is ' + answerText() + '.'; $('#m7check').disabled = true;
        if (q.kind === 'mc') U.$$('#m7answer .opt').forEach(function (o, i) { o.classList.toggle('is-right', i === q.correct); }); }
      $('#m7sol').hidden = false;
    });
    $('#m7next').addEventListener('click', next);
    $('#m7reset').addEventListener('click', function () { if (confirm('Reset your practice record?')) { st = { total: 0, correct: 0, streak: 0, best: 0, n: 0, off: st.off, topics: {} }; save(st); score(); } });
    score(); next();
  });
})();
