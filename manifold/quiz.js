/* ============================================================================
   Manifold Learning Playground — section 6, the quiz.
   Every option carries its own explanation, shown once you answer.
   ========================================================================= */

(function () {
  'use strict';
  var U = window.UI, $ = U.$, $$ = U.$$;
  function t(s) { return '<span class="tex" data-tex="' + s.replace(/"/g, '&quot;') + '"></span>'; }

  var Q = [
    { topic: 'Intrinsic vs. ambient', q: 'You photograph one face at 64 × 64 grayscale. Between shots, the person only turns their head left and right, and a single lamp moves anywhere on a hemisphere around them. What are the ambient and intrinsic dimensions?',
      opts: [
        ['Ambient 4096, intrinsic 3', true, 'Correct. Each photo is a vector of 64 × 64 = 4096 pixels, so that is the ambient space. Only three numbers generate the photos: head angle plus two angles for the lamp on its hemisphere. So the photos lie on a 3-dimensional manifold.'],
        ['Ambient 3, intrinsic 4096', false, 'This is backwards. The ambient space is where the data <i>lives</i> (4096 pixel values). The intrinsic dimension counts the degrees of freedom that <i>generate</i> it.'],
        ['Ambient 4096, intrinsic 4096', false, 'Only if every pixel could vary independently. Here pixels move together because a few physical variables drive all of them, as with the digit sliders in section 2.'],
        ['Ambient 64, intrinsic 2', false, 'The image has 64 rows, but each one holds 64 pixels: 4096 numbers in all. The lamp alone gives 2 degrees of freedom, but the head rotation adds a third.']] },
    { topic: 'Intrinsic vs. ambient', q: 'The Swiss roll sits in ℝ³. Which statement is right?',
      opts: [
        ['Its intrinsic dimension is 3, because the points need three coordinates.', false, 'Needing 3 coordinates to <i>write down</i> a point is the ambient dimension. Near any point the roll looks like a flat sheet, which is 2-dimensional.'],
        ['Intrinsic 2, ambient 3: locally it is a sheet, embedded in 3-D space.', true, 'Correct. Two numbers, position along the roll and height, pin down a point on the roll. The extra ambient dimension is spent on the bending.'],
        ['Intrinsic 1, because it is a spiral.', false, 'The spiral is only its cross-section. The roll also extends along the height axis, so it is a 2-D sheet.'],
        ['It has no well-defined dimension because it is curved.', false, 'Curvature does not change dimension. A manifold has a dimension because every small patch looks like ℝ<sup>L</sup>, whatever its global shape.']] },
    { topic: 'Manifold hypothesis', q: 'What does the manifold hypothesis claim?',
      opts: [
        ['Every dataset lies exactly on a smooth low-dimensional manifold.', false, 'Too strong. The hypothesis is about natural high-dimensional data <i>concentrating near</i> such a manifold, with noise around it. It is a useful assumption, not a theorem about all data.'],
        ['Natural high-dimensional data concentrate near a much lower-dimensional manifold in the ambient space.', true, 'Correct (Murphy §20.4.2). Real images, sounds and similar data have far fewer real degrees of freedom than raw coordinates. This makes learning possible despite the curse of dimensionality.'],
        ['High-dimensional data are approximately Gaussian.', false, 'That is a distributional assumption, and it is often wrong. A Gaussian puts mass everywhere in ℝ<sup>D</sup> and is not concentrated near a thin curved set.'],
        ['The ambient dimension of images is small.', false, 'The ambient dimension of images is large (784 for MNIST). The hypothesis says the <i>intrinsic</i> dimension is small.']] },
    { topic: 'Manifold hypothesis', q: 'In section 2, a uniformly random vector in [0,1]⁷⁸⁴ looks like static, never like a digit. What does this suggest?',
      opts: [
        ['Digits fill a vanishingly small, structured part of pixel space.', true, 'Correct. If digits filled the space, random draws would sometimes look like them. They never do, so the set of digits is thin. That is evidence for the manifold hypothesis.'],
        ['784 dimensions is not enough to represent digits.', false, 'Every digit image is exactly a point in [0,1]⁷⁸⁴, so 784 dimensions represent them perfectly. The point is that they use only a tiny part of that space.'],
        ['The uniform distribution is the right model for images.', false, 'It is the opposite. Uniform samples look nothing like images, which shows how far real images are from uniform.'],
        ['Digits are linearly separable.', false, 'That is a statement about classification, not about where the data sit. The experiment shows that the data are concentrated.']] },
    { topic: 'Tangent spaces', q: 'For a 2-D manifold ℳ ⊂ ℝ³, what is the tangent space ' + t('T_p\\mathcal M') + ' at a point p?',
      opts: [
        ['The 2-D vector space of velocities of curves passing through p', true, 'Correct. ' + t('T_p\\mathcal M') + ' collects every possible direction of motion within ℳ at p. For a surface it is spanned by ' + t('\\partial f/\\partial u') + ' and ' + t('\\partial f/\\partial v') + ' (the blue and violet arrows in section 1). Its dimension equals the manifold&rsquo;s.'],
        ['The whole of ℝ³', false, 'That is the ambient space. Directions out of the surface (the normal) are excluded from ' + t('T_p\\mathcal M') + '.'],
        ['The normal line at p', false, 'The normal line is the orthogonal complement of the tangent space, the one direction <i>not</i> in it.'],
        ['The set of points on ℳ within distance r of p', false, 'That is a neighborhood, a set of points on the manifold. The tangent space is a vector space of directions, and it is flat even when the neighborhood curves.']] },
    { topic: 'Riemannian manifolds', q: 'What makes a manifold <b>Riemannian</b>?',
      opts: [
        ['A smoothly varying inner product ' + t('g_p') + ' on every tangent space', true, 'Correct. The metric tensor lets you measure lengths and angles of tangent vectors, so you can compute curve lengths ' + t('\\int\\!\\sqrt{g(\\dot\\gamma,\\dot\\gamma)}\\,dt') + ' and, from those, geodesic distances. Section 1 shows g at the chosen point.'],
        ['Being embedded in some Euclidean space', false, 'Many manifolds are embedded, but the Riemannian structure is the metric itself. The metric can be defined with no ambient space at all.'],
        ['Having zero curvature everywhere', false, 'Zero curvature makes a manifold <i>flat</i>. Spheres and saddles are Riemannian and curved.'],
        ['Being compact', false, 'Compactness is a topological property and unrelated. The plane is Riemannian and not compact.']] },
    { topic: 'Riemannian manifolds', q: 'The Swiss roll has Gaussian curvature K = 0, and the sphere has K > 0. What follows?',
      opts: [
        ['The roll can be unrolled into the plane with no stretching; the sphere cannot.', true, 'Correct. K is intrinsic (Gauss&rsquo;s <i>Theorema Egregium</i>), and an isometry must preserve it. The roll is bent paper, so Isomap can recover it exactly as N → ∞. No 2-D embedding of the sphere keeps all geodesic distances.'],
        ['Both can be flattened isometrically if you try hard enough.', false, 'No method can do this for the sphere. Its positive curvature is intrinsic, and every flat map distorts it, as world maps do.'],
        ['K = 0 means the roll is a plane in ℝ³.', false, 'The roll is clearly bent in ℝ³: its extrinsic curvature is nonzero. K = 0 says it is <i>intrinsically</i> flat. One principal curvature is 0, along the height axis.'],
        ['Curvature only matters for t-SNE.', false, 'Curvature limits every method that tries to preserve distances, such as Isomap and MDS.']] },
    { topic: 'Geodesic vs. Euclidean', q: 'Two points sit on adjacent layers of the Swiss roll, one directly above the other in ℝ³. Which is true?',
      opts: [
        ['Euclidean distance is small; geodesic distance is large.', true, 'Correct. The chord crosses the gap in one step, while a walk along the sheet goes a full turn around the roll. The "Neighbors across a layer" button in section 3 shows a ratio of about 5–10×.'],
        ['Both distances are about equal.', false, 'They agree only for nearby points on the same sheet. That is why Isomap builds the geodesic from short hops.'],
        ['Geodesic is shorter than Euclidean.', false, 'This can never happen. The straight line is the shortest curve in ℝ³, so any curve restricted to the manifold is at least as long.'],
        ['PCA sees the geodesic distance.', false, 'PCA uses only Euclidean geometry (covariance), so it treats these points as close. That is why it stacks layers on top of each other.']] },
    { topic: 'Geodesic vs. Euclidean', q: 'In Isomap, what happens if k (the number of neighbors) is too large on the Swiss roll?',
      opts: [
        ['Edges jump between layers ("short circuits"), and the graph distances collapse toward Euclidean ones.', true, 'Correct. One bad edge is a shortcut that many shortest paths will use, so a few errors corrupt the whole distance matrix and the embedding folds. Try k = 22 in section 4, or show the graph in section 3.'],
        ['The graph disconnects.', false, 'That happens when k is too <i>small</i>. Larger k adds edges and connects more.'],
        ['Nothing. More neighbors always means better geodesics.', false, 'That holds only while all neighbors are on the same local sheet. Past the layer spacing, extra neighbors are wrong neighbors.'],
        ['The algorithm becomes linear, like PCA.', false, 'With k = N − 1 the graph is complete and graph distances equal Euclidean ones, so it reduces to classical MDS, which equals PCA. The degradation along the way is through short circuits, which is the answer above.']] },
    { topic: 'Linear vs. nonlinear', q: 'Why can no linear method (PCA, or any linear projection ℝ³ → ℝ²) unroll the Swiss roll?',
      opts: [
        ['A linear map sends straight-line chords to chords, so points that are close in ℝ³ (adjacent layers) stay close, and the layers overlap.', true, 'Correct. Linear maps only rotate, scale and drop axes. Unrolling requires a different "stretch" at each point along the spiral, which is a nonlinear map.'],
        ['PCA is too slow on 3-D data.', false, 'PCA on 3-D data is instant. The problem is the geometry, not the speed.'],
        ['PCA needs labels.', false, 'PCA is unsupervised. It uses only the covariance of x.'],
        ['It could, if we kept 3 components.', false, 'Keeping all 3 components just rotates the roll. It stays rolled up.']] },
    { topic: 'Parametric vs. nonparametric', q: 'You embedded 10,000 images with t-SNE. Fifty new images arrive. Which is true?',
      opts: [
        ['t-SNE learned no mapping f(x); you must re-run it, or use an out-of-sample extension or parametric t-SNE.', true, 'Correct. t-SNE optimizes one free vector ' + t('\\mathbf z_n') + ' per training point; it is nonparametric. PCA (a parametric method) would just compute ' + t('\\mathbf W^\\top(\\mathbf x-\\boldsymbol\\mu)') + '.'],
        ['Apply the learned weight matrix to the new images.', false, 't-SNE has no weight matrix. Its only outputs are the coordinates of the training points.'],
        ['Run gradient descent on the new points only, with the old ones fixed. This is exactly t-SNE.', false, 'This is a reasonable heuristic, but it is not the original objective and its results differ from a full re-run. It is a kind of out-of-sample extension.'],
        ['Use Isomap\'s mapping instead.', false, 'Isomap is also nonparametric, so it has the same problem. It needs Nyström-style extensions.']] },
    { topic: 'Parametric vs. nonparametric', q: 'Which of these is a <b>parametric</b> dimensionality reduction method?',
      opts: [
        ['PCA', true, 'Correct. PCA learns ' + t('\\mathbf W \\in \\mathbb R^{D\\times L}') + ' and ' + t('\\boldsymbol\\mu') + ', a fixed-size function that embeds any new x. Autoencoders and parametric t-SNE are also parametric.'],
        ['Isomap', false, 'Isomap returns coordinates only for the graph&rsquo;s nodes.'],
        ['LLE', false, 'LLE solves for one z per training point (bottom eigenvectors of M), so there is no reusable map.'],
        ['Laplacian eigenmaps', false, 'Laplacian eigenmaps solves an eigenproblem over training points and has no function to apply to new data.']] },
    { topic: 'Algorithm differences', q: 'Which method finds the <b>bottom</b> eigenvectors of ' + t('(\\mathbf I-\\mathbf W)^\\top(\\mathbf I-\\mathbf W)') + ', where W holds local reconstruction weights?',
      opts: [
        ['LLE', true, 'Correct. Step 1 fits each point as an affine combination of its neighbors. Step 2 finds Z that the same weights rebuild best, which minimizes ' + t('\\operatorname{tr}(\\mathbf Z^\\top\\mathbf M\\mathbf Z)') + '. Skip the constant eigenvector.'],
        ['Isomap', false, 'Isomap takes the <i>top</i> eigenvectors of the double-centered squared geodesic distance matrix (classical MDS).'],
        ['Laplacian eigenmaps', false, 'Close: it also takes bottom eigenvectors, but of the graph Laplacian L = D − W, using heat-kernel similarities rather than reconstruction weights.'],
        ['Kernel PCA', false, 'Kernel PCA takes the <i>top</i> eigenvectors of the centered kernel matrix.']] },
    { topic: 'Algorithm differences', q: 'Isomap is best described as…',
      opts: [
        ['classical MDS applied to shortest-path distances on a kNN graph', true, 'Correct. Step 1 builds the graph, step 2 runs Dijkstra or Floyd for d<sub>G</sub>, step 3 applies MDS to d<sub>G</sub>. Fed plain Euclidean distances, step 3 gives PCA.'],
        ['PCA with an RBF kernel', false, 'That is kernel PCA. (Isomap <i>can</i> be written as kernel PCA with the kernel −½Hd<sub>G</sub>²H, but that kernel is built from geodesics, not an RBF.)'],
        ['t-SNE with a Gaussian in the embedding', false, 'That is (symmetric) SNE. t-SNE&rsquo;s fix was the heavy-tailed Student-t.'],
        ['a neural network trained to reconstruct x', false, 'That is an autoencoder (§20.3). It is parametric, while Isomap is not.']] },
    { topic: 'Algorithm differences', q: 'In a t-SNE plot, cluster A is three times as far from B as from C. What can you conclude?',
      opts: [
        ['Not much. t-SNE preserves local neighborhoods, not global distances or gaps.', true, 'Correct. The heavy tails exaggerate gaps, and the per-point σ<sub>i</sub> normalizes density, so cluster sizes and between-cluster distances are unreliable. Compare distance corr. with 10-NN kept for t-SNE in section 4.'],
        ['A is three times as dissimilar to B as to C.', false, 'That would need distances to be preserved, which t-SNE does not attempt. Its KL objective barely penalizes moving far-apart points around.'],
        ['B is a larger cluster.', false, 'Cluster <i>size</i> is not meaningful in t-SNE either. Dense and sparse clusters come out about the same size.'],
        ['The perplexity is too high.', false, 'The perplexity changes the picture, but no setting makes global distances trustworthy.']] },
    { topic: 'Algorithm differences', q: 'Laplacian eigenmaps minimizes ' + t('\\sum_{ij} W_{ij}\\lVert\\mathbf z_i-\\mathbf z_j\\rVert^2') + '. Why does it need the constraint ' + t('\\mathbf Z^\\top\\mathbf D\\mathbf Z = \\mathbf I') + '?',
      opts: [
        ['Without it, putting every point at the same location gives the minimum value, 0.', true, 'Correct. The objective only <i>rewards</i> closeness, so collapse is optimal. The scale constraint rules this out, and ' + t('\\mathbf Z^\\top\\mathbf D\\mathbf 1=\\mathbf 0') + ' removes the trivial constant solution (λ<sub>0</sub> = 0).'],
        ['To make the problem convex', false, 'It actually makes the problem nonconvex in general, but it becomes a generalized eigenproblem that is solved exactly.'],
        ['To preserve geodesic distances', false, 'Laplacian eigenmaps never preserves distances, only locality.'],
        ['To speed up the eigensolver', false, 'The constraint defines the problem. It is not a numerical trick.']] }
  ];

  /* Options are written correct-first above; show them in a fixed shuffled
     order per question so the right answer isn't always A. */
  function order(qi) {
    var r = window.ML.rng(1000 + 37 * qi), a = [0, 1, 2, 3];
    for (var i = 3; i > 0; i--) { var j = Math.floor(r() * (i + 1)), t = a[i]; a[i] = a[j]; a[j] = t; }
    return a;
  }
  var root = $('#quiz'), answered = {}, score = 0;
  function render() {
    root.innerHTML = '';
    Q.forEach(function (item, qi) {
      var card = document.createElement('section'); card.className = 'qcard';
      card.innerHTML = '<div class="q-meta"><span class="q-num">' + (qi + 1) + '</span><span class="q-topic">' + item.topic + '</span></div>' +
        '<p class="q">' + item.q + '</p><div class="opts"></div><div class="q-fb" aria-live="polite"></div>';
      var box = $('.opts', card);
      order(qi).forEach(function (oi, pos) {
        var o = item.opts[oi], b = document.createElement('button'); b.type = 'button'; b.className = 'opt'; b.dataset.oi = oi;
        b.innerHTML = '<span class="opt-letter">' + 'ABCD'[pos] + '</span><span class="opt-text">' + o[0] + '</span><span class="opt-why"></span>';
        b.addEventListener('click', function () { answer(qi, oi, card); });
        box.appendChild(b);
      });
      root.appendChild(card);
    });
    U.texIn(root);
    $('#q-score').textContent = '0 / 0';
  }
  function answer(qi, oi, card) {
    if (answered[qi] != null) return;
    answered[qi] = oi;
    var item = Q[qi], ok = item.opts[oi][1];
    if (ok) score++;
    $$('.opt', card).forEach(function (b) {
      var j = +b.dataset.oi, o = item.opts[j];
      b.disabled = true;
      b.classList.add(o[1] ? 'is-right' : j === oi ? 'is-wrong' : 'is-other');
      if (j === oi) b.classList.add('chosen');
      var why = $('.opt-why', b); why.innerHTML = o[2];
    });
    var fb = $('.q-fb', card);
    fb.className = 'q-fb ' + (ok ? 'ok' : 'no');
    fb.textContent = ok ? 'Correct. Every option is explained below.' : 'Not quite. The correct answer is marked in green, and every option is explained.';
    U.texIn(card);
    $('#q-score').textContent = score + ' / ' + Object.keys(answered).length + (Object.keys(answered).length === Q.length ? '  · done' : '');
  }
  $('#q-reset').addEventListener('click', function () { answered = {}; score = 0; render(); });
  render();
})();
