# Playbook: building a new "by hand" breakdown

This is the recipe behind **Attention, by hand** (`attention/`). Follow it, or hand it to Claude, to build
the next architecture breakdown so it looks, flows and teaches the same way, and lands in this repo next
to the others.

A breakdown is one self-contained web page that walks through **one full forward pass** of an
architecture, every matrix, every shape, every sum, with a practice panel beside each step, plus take-home
material (notes, a self-checking Jupyter notebook and a printable worksheet).

> **The promise to the reader.** Finish the breakdown and its homework and you can close your eyes and
> picture how the architecture works. Every later variant then reads as a small edit to that picture.
> Every decision below serves that promise.

---

## 0. Quick start

1. Pick the architecture and its paper. Choose a short folder name (the *slug*), e.g. `bert`, `vit`,
   `gpt2`, `mamba`, `diffusion`.
2. Do the learning first (section 3): walkthrough notes with shapes, then every calculation by hand.
3. Copy `attention/index.html` to `<slug>/index.html` and rebuild its content (sections 4 to 7). Keep the
   shell, the CSS tokens, the components and the practice engine. Replace the chapters, the figure, the
   maths and the citation.
4. Write the notebook generator, build the notebook and embed it (section 8).
5. Adapt the worksheet generator (section 9).
6. Add a card to the hub `index.html` and a row to the README table (section 10).
7. Push on a branch, open a PR, merge to `main`; GitHub Pages publishes it (section 11).
8. Run the QA checklist (section 12).

To hand the whole job to Claude, use the prompt in section 13.

---

## 1. Where things live

```
llmbreakdown/                      repo: github.com/GuptaAditya95/llmbreakdown
├── index.html                     the hub ("Let's break it down"), one card per breakdown
├── README.md                      Breakdowns table, inspiration, star request
├── PLAYBOOK.md                    this file
├── attention/                     breakdown 01
│   ├── index.html                 the interactive site (single file, ~300 KB)
│   ├── notes/                     walkthroughs, calculations, notebook generators, notebooks
│   └── worksheets/                a sample printed worksheet (PDF)
└── <slug>/                        your next breakdown, same shape
```

Live URLs (GitHub Pages, branch `main`, folder `/ (root)`):

- Hub: `https://guptaaditya95.github.io/llmbreakdown/`
- A breakdown: `https://guptaaditya95.github.io/llmbreakdown/<slug>/`

If the `letsbreakitdown` organisation move happens, see section 11.3. Links inside the pages are all
relative (`../` back to the hub, `#encoder` between parts), so they keep working after a move.

In the Claude project, working copies sit in the shared project folder under
`github-pages/llmbreakdown/` (a mirror of the repo) with earlier drafts in `notes/`, `site/` and
`worksheets/`.

---

## 2. Principles (from Aditya's feedback on the first breakdown)

These came out of reviewing Attention, by hand. Treat them as rules.

1. **Say who it's for.** Someone who enjoys learning architectures from scratch. No prior training
   experience needed, only comfort with multiplying and adding and curiosity about *why* each step exists.
2. **Say why you made it.** Building the breakdown is how the author learns, and the reader is invited to
   build their own, more vivid version.
3. **One set of sizes, start to finish.** Never switch from paper sizes to toy sizes mid-pass. Run the
   whole forward pass at small *recommended* sizes (same proportions as the paper) and show the paper's
   sizes next to every shape. Let the reader change the sizes, but recommend the defaults to first-timers.
4. **A practice panel beside every step.** The reader fills in matrix entries, shapes and concept
   questions themselves, and checks them.
5. **Mobile-first practice.** On narrow screens the practice panel opens as a pop-up sheet and closes back
   to the walkthrough, so the flow isn't interrupted by long scrolls.
6. **Never cramp the numbers.** Vectors and matrices keep their spacing on phones; they scroll sideways
   in their own box instead of squashing.
7. **Linger on the beautiful idea.** Each architecture has one idea that makes it click (for attention it
   was sinusoidal positional encoding). Give it extra room: ask the reader how *they* would solve the
   problem, compare candidate designs, then reveal the paper's choice and where the idea went next.
8. **Be exact about dimensions.** Every step states its output shape for this run *and* for the paper.
9. **Track shapes visually.** A sticky header shows a dot per step and a shape bar showing the output
   size after every step, so you can watch where the matrix grows, shrinks and stays the same.
10. **End each part with a full-pass homework.** A downloadable notebook where you fill in every matrix
    for a tiny example and each one is checked as you go.

---

## 3. Phase 1: learn it and write the notes first

Before any HTML, produce the material the site will be built from. These files go in `<slug>/notes/`
and double as downloadable references. Attention's equivalents are named in brackets.

| File | What it holds | Attention example |
|---|---|---|
| `<part>-walkthrough.md` | Plain-language walk through one example, step by step, with a symbol table (symbol, meaning, paper value) and the shape after every step, citing paper sections (§3.2.1, Eq. 1). Ends with "learned parameters per layer" and self-check questions. | `encoder-walkthrough.md`, `decoder-walkthrough.md` |
| `<topic>-calculations.md` | Every number for one layer at small sizes: inputs, each product, each softmax, each norm. This is your answer key for the site. | `encoder-layer-calculations.md`, `positional-encoding-calculations.md` |
| `<topic>_example.py` | A tiny NumPy script that reproduces the calculations. Use it to verify the markdown. | `encoder_layer_example.py` |
| optional data | A CSV or table of anything too large to print, at paper size. | `positional-encoding-6x512.csv` |

Rules for the notes:

- Shapes are written `(rows × columns)`, one row per token (or patch, pixel, timestep). Mention that real
  training adds a batch dimension in front.
- Cite the paper section on every claim. Use the arXiv version's numbering and say so.
- Separate what the paper says from common interpretations ("The paper doesn't say why. A common
  reading is...").
- Weights are made-up stand-ins for learned values. Say so once, clearly.

---

## 4. Phase 2: the page

### 4.1 Start from the existing file

Copy `attention/index.html`. It is one self-contained file: inline CSS, inline JS, Google Fonts as the
only external request, notebooks embedded as base64. Keep it that way; it makes GitHub Pages hosting,
offline use and Claude artifacts all work.

Keep: the `:root` tokens, base components, the hash router, the sticky header (dots + shape bar), the
settings panel, `matrix()`, the maths helpers, the seeded RNG, the practice engine (`buildPractice`,
`fillPractice`, the sheet), the notebook download handler and the worksheet shell.

Replace: title and meta description, the cover text and figure, the citation strings (`CITES`), the
`PAPER` and `RECOMMENDED` sizes, the chapters, the render functions and the `SHAPES` table.

### 4.2 Design system

**The look:** a squared school notebook. Off-white paper with a faint blue grid, ink-blue text,
handwritten headings, white cards held on by a strip of yellow tape, a highlighter for key phrases,
red-pen accents. It must feel hand-made and warm, never like a dashboard.

**Fonts** (one Google Fonts link):

- `Caveat` 500/700 for headings, step numbers and big hand-written values (`--font-hand`)
- `Patrick Hand` for notes, captions, buttons, shape notes (`--font-note`)
- `Lexend` 300/400/500/600 for body text (`--font-body`), body weight 300
- `JetBrains Mono` 400/600 for numbers, shapes and code (`--font-mono`), with tabular figures

**Colour tokens** (copy exactly; the hub uses the same ones):

| Token | Light | Dark | Used for |
|---|---|---|---|
| `--paper` | `#fbfaf3` | `#1a1f27` | page background |
| `--grid-line` | `rgba(64,112,196,.13)` | `rgba(150,185,240,.075)` | 24px squared grid |
| `--margin-line` | `rgba(214,72,72,.45)` | `rgba(240,110,110,.33)` | red notebook margin |
| `--ink` / `--ink-soft` / `--pencil` | `#1c2848` / `#4a5574` / `#7d8398` | `#eceade` / `#b8bccb` / `#8a90a4` | text levels |
| `--card` / `--card-edge` | `#fff` / `rgba(28,40,72,.16)` | `#222933` / `rgba(255,255,255,.11)` | cards |
| `--hl` | `rgba(255,232,84,.78)` | `rgba(255,222,80,.30)` | highlighter, sticky notes |
| `--tape` | `rgba(236,196,92,.55)` | `rgba(236,196,92,.30)` | tape on cards |
| `--pos-rgb` | `212 66 44` (red) | `255 118 92` | positive values, **paper** sizes, accents |
| `--neg-rgb` | `38 94 196` (blue) | `108 158 255` | negative values, **this run** sizes, focus ring |
| `--ok` | `#2d8a4e` | `#5fcf86` | solved |
| role colours | `--q #d4711a`, `--k #11897a`, `--v #7346bd` | lighter versions | one colour per role |

Role colours: give each named actor in the architecture its own colour and use it in text, matrices and
the figure. For attention that was Q, K and V. For a ViT it might be patch, position and class token;
for Mamba, A, B, C and Δ.

Dark mode: define the dark values under `@media (prefers-color-scheme: dark)` with
`:root:not([data-theme="light"])`, and again under `:root[data-theme="dark"]`. Set `color-scheme`.

**Components** (all in the CSS already):

| Class | What it is |
|---|---|
| `.btn` (`.big`, `.small`, `.ghost`) | hand-drawn button: wobbly border-radius, 3px ink offset shadow, nudges on hover |
| `.chip` | pill toggle; `aria-pressed="true"` fills it with highlighter |
| `.card.tape` | white card with tape on top; holds every worked example |
| `.paper-note` / `.paper-note.sticky` | side notes on the cover; `.sticky` is a yellow sticky note |
| `.hl` | highlighter stroke behind a phrase. Use once or twice per step, for the phrase to remember |
| `.label` | small uppercase grey label ("Part 1 · Encoder", "Notice") |
| `.why` | highlighter-edged aside for "Notice", "Why", "Where the idea went next" |
| `.shape-note` | dashed box stating the output shape: `<span class="mono run">` this run and `<span class="mono paperdim">` paper |
| `.sec-ref` | pencil-grey `§3.4` reference to the paper |
| `.run` / `.paperdim` | blue for this run, red for the paper. Never swap them |
| `.note` | handwritten aside in Patrick Hand |
| `.practice` | the "Your turn" sheet: lined paper, ink border, offset shadow |
| `.homework-final` | dashed box at the end of a part with the notebook download |

**Layout:** `.wrap` max 1240px with 16px side gutters (safe-area aware). A step is a two-column grid,
text `minmax(0,40ch)` on the left and the sticky `.viz` card on the right; under 980px it becomes one
column and the practice panel turns into the pop-up sheet. Under 560px the header hides the title and
shrinks the dots. Wide matrices sit in `.scroll-x`. Respect `prefers-reduced-motion` for every animation
and for smooth scrolling.

### 4.3 Page structure

One HTML file, a few "pages" switched by the URL hash (`#home`, `#<part1>`, `#<part2>`, …), plus deep links
to any step (`#tokens`, `#embed`). Attention has encoder, decoder and inference parts; pick the natural halves of
your architecture (one part is fine for a decoder-only model, three for an encoder, a bridge and a head). A
last part on **how the trained model is used** (inference: the generation loop, caching, decoding strategies)
is worth adding to any generative architecture; Attention's Part 3 is the template. Add every new page id to
`route()` and to `PAGES`.

**Cover (`#home`)**, in this order:

1. Kicker: `<a href="../">Let's break it down</a> · An interactive notebook on <Authors> (<year>)`.
   This is the link back to the hub; always keep it.
2. `h1`: "<Thing>, *by hand*" with the hand-drawn red SVG underline under "by hand".
3. Lede: which paper, who wrote it, and the promise: type an input, watch every number computed with the
   shapes beside it, then do the sums yourself.
4. Parts cards ("Part 1 · The encoder", "Enter →").
5. The figure: the paper's main architecture figure **redrawn by hand** as inline SVG (Caveat labels,
   `.box` and `.wire` classes, draw-on animation). Each half is a link into its part, with a red hand
   annotation ("click the encoder ↓"). Caption: "Redrawn by hand from Figure N of ...".
6. "Who this is for" (with the red margin line) beside "How the notebook works" (Read, Watch, Do the
   homework, Track the shapes).
7. Three notes: a yellow sticky "Is this how ChatGPT works?" style note that connects the architecture to
   what readers use today; "Cite the paper" with APA 7 / IEEE / BibTeX tabs and a copy button; "Thank you"
   for the resources that shaped your understanding.
8. Footer: "Made with Claude. This site was built with Claude by Anthropic. The vision, structure and
   ideas are entirely my own." and "Weights in the worked examples are random stand-ins for learned
   values."

**Each part page:**

1. Sticky header: `← cover`, the part title, step dots (one per step; a green ring when its practice is
   fully solved), the current shape in run and paper sizes, and the shape bar.
2. Intro: "One set of sizes, start to finish". A settings table with each hyperparameter, its paper value
   in red, a stepper for this run in blue, a `?` tooltip explaining the choice and the paper's ablation
   evidence (Table 3 rows), and "Reset to recommended". Note for first-timers to keep the defaults.
3. Chapters (`section.chapter`, "Chapter N" label + handwritten h2), each holding steps.
4. The last step of the part shows the whole journey (a table of every step and its shape) and the
   **Final homework** box.
5. After the last part: "Make your own worksheet".

**Each step** (`article.step` with `data-step="<id>"` and `data-name="<shape label>"`):

```html
<article class="step" data-step="scores" data-name="S = QKᵀ">
  <div class="txt">
    <div class="eyebrow"><span class="stepno">9</span><h3>Score every pair of tokens</h3></div>
    <p>Plain language first: what happens and why.</p>
    <p data-tpl>Then the exact maths, using live sizes like <span class="run">{n} × {n}</span> <span class="sec-ref">§3.2.1</span>.</p>
    <div class="why"><span class="label">Notice</span>One observation to carry forward.</div>
    <div class="shape-note" data-tpl>Q ({n} × {dk}) · Kᵀ ({dk} × {n}) → S: <span class="mono run">{n} × {n}</span> · paper <span class="mono paperdim">{n} × {n}</span>.</div>
  </div>
  <div class="viz"><div class="card tape"><p class="card-cap">caption</p><div id="vizScores"></div></div></div>
</article>
```

`data-tpl` elements are re-filled with the current sizes (`{n}`, `{d}`, `{h}`, `{dk}`, `{dff}`, `{N}`,
...) whenever the settings change. Add a `SHAPES` entry per step so the header and shape bar can show it.

---

## 5. Phase 3: the intuitive flow

How to turn the notes into a sequence of steps that builds a picture rather than a list of formulas.

1. **Start from the reader's own input.** Step 1 is always "type your own X" (a sentence, a tiny image,
   a sequence of numbers) with a few examples to click, including one that shows off the idea ("dog bites
   man" vs "man bites dog"). Everything after is computed live from it.
2. **One step = one operation = one shape change.** If a step changes the shape twice, split it.
3. **Plain words, then exact maths, then the shape.** Every step's text follows that order.
4. **Ask before you tell.** For the key idea, pose the problem first ("How could you write position into a
   vector?"), list the properties a good answer needs, let the reader flip through candidate designs on
   the card, then reveal the paper's choice as the one that ticks every box.
5. **Plant and pay off.** Use `Notice` boxes to point at something that matters later ("the same word
   always gets the same row... keep that in mind for chapter 2"), and pay it off when you get there.
6. **Paper vs. interpretation.** When the paper gives no reason, say so, then give the common reading.
7. **Connect forward.** A "Where the idea went next" aside on the big ideas (e.g. sinusoidal PE → relative
   positions → RoPE) and the cover note on how today's models differ.
8. **Animate the arithmetic.** The worked example card highlights the row and column being multiplied and
   shows the products being summed ("6 products added together"). The reader should be able to *watch*
   one entry being made.
9. **Repeat by summary, not by re-deriving.** For stacked layers, one "Repeat N times" step that shows
   what changes (the values) and what doesn't (the shape).
10. **Close each part with the full picture:** the journey table, then the final homework.

Suggested chapter skeleton for most architectures:

| Chapter | Steps |
|---|---|
| Input to vectors | your input, tokenize/patchify, embed, scale |
| Position or order (if any) | why it's needed, design candidates, the paper's scheme, build it, add it |
| The core mixing operation | project, score, normalise, combine, merge heads, add & norm |
| The per-position network | expand, nonlinearity, compress, add & norm |
| Stacking | repeat N times, the output |
| The head | to logits / classes / pixels, softmax, how training vs. inference differ |
| Using it (inference) | encode once, the generation loop one turn at a time, stopping, caching, decoding strategies (greedy, beam, sampling) |

---

## 6. Phase 4: "Your turn" exercises on every step

Every step that computes something gets a practice panel. It sits under the worked example on desktop and
opens as a pop-up sheet from a "Practice this step (k)" button on screens under 980px.

Register it inside the step's render function, so its numbers come from the reader's current input:

```js
buildPractice('embed', () => { const i = rint(S.n), j = rint(c.d); return [
  {type: 'num', q: `Look up E[“${S.toks[i]}”, d${j}].`,
   data: `row ${S.ids[i]} of the table = ${vecStr(S.E[i])}`,
   ans: r2(S.E[i][j]), tol: .011,
   hint: `Count along to column d${j} (the first column is d0).`,
   show: `optional working shown before the answer`},
  {type: 'shape', q: 'What shape is E in the paper?', ans: [S.n, 512], hint: 'rows = tokens, columns = d_model'},
  {type: 'choice', q: 'Two copies of the same word get…', options: ['the same row', 'different rows'], ans: 0,
   why: 'The lookup only sees the token. Position is added later.'}
]; });
```

Task types: `num` (a number, default tolerance 0.02, `tol: 0` for integers, `fmt` to print the answer),
`shape` (any "6 × 8" / "6x8" / "(6, 8)" form), `choice` (with `why` shown on success). Each task gets
Check, Hint and Show answer; the panel has "New numbers" and a solved counter, and a fully solved step
puts a green ring on its dot.

A good set per step is **two or three tasks**:

- one *read or lookup* task that teaches where a number comes from,
- one *compute one entry* task (one dot product, one softmax value, one normalised value), giving the
  vectors in `data` so it's doable by hand,
- one *shape* task (often the paper's size) or a *concept* choice about why the step exists.

Keep the arithmetic humane: one or two decimals, short vectors, and a hint that names the exact row and
column. Accept small rounding differences (`tolOf(v) = 0.03 + 0.01·|v|` for chained results).

Numbers must be reproducible: weights come from `mulberry(seed)` with a seed built from the layer and the
sizes, inputs are hashed with `hashStr`, and values are rounded with `r1`/`r2` at every stage so the
on-screen numbers are exactly the ones the reader should use.

---

## 7. Phase 5: the worked-example cards

The `.viz` card for each step is drawn by a `render<Step>()` function that writes into its `#viz<Step>`
div using `matrix(M, {title, rowL, colL, dim})`. Matrix cells are tinted red for positive and blue for
negative, scaled by magnitude. Use `setAnims(step, [...])` to register animations that run when the step
scrolls into view (and are skipped under reduced motion). Keep every card readable at 360px wide: put
matrices in `.scroll-x`, cap the input length (attention caps at 8 words), and use `compact` mode above
12 columns.

---

## 8. Phase 6: the final-homework notebook

Each part ends with a downloadable Jupyter notebook in which the reader fills in **every matrix** of one
full pass for a tiny example, and each is checked as they go. Attention ships two:
`encoder-practice.ipynb` (encoder only) and `transformer-practice.ipynb` (encoder + decoder).

Write a generator, `notes/make_<slug>_practice.py`, modelled on `make_encoder_practice.py`:

- **`HELPERS` cell** (run first): the settings block with paper values in comments, a tiny vocabulary,
  hand-picked weights with simple values (0, ±0.5, ±1) so the arithmetic is doable, and the checker.
  Tell the reader not to peek at `REF`.
- **`_RULES`**: one function per matrix name that computes the expected value from the reader's *own*
  previously checked matrices (`got(k)`), so rounding never snowballs.
- **`check(name, yours)`**: verifies shape, then every entry with tolerance `0.02 + 0.01·|exp|`, lists up
  to six wrong entries and gives a specific hint for the first (`name[r,c] = row r of A · column c of B =
  ...`). Blank (all `nan`) cells get a gentle nudge.
- **`reveal(name)`**: prints the answer and lets the reader carry on.
- **`check_shapes(answers)`**: a final cell where the reader writes the paper-size shapes.
- **Cells**: an intro markdown (how it works + a sizes table: here vs. paper), then for each step a short
  markdown (what to do, the shape) followed by a `fill(name, rows, cols, comment)` cell full of `nan`.
  End with "Want more?" (change the input, edit the weights, compare with the site).
- Give each cell a stable id (`cell-00`, ...) and tag fill cells `fill:<name>`.

Build and verify:

```bash
python3 notes/make_<slug>_practice.py notes/<slug>-practice.ipynb
# verify: fill every cell with the reference answer and run it top to bottom; every check must print ✓
```

Embed it in the page so the download works offline and inside Claude artifacts:

```bash
cd <slug>/notes && zip -j /tmp/<slug>-practice.zip <slug>-practice.ipynb
base64 -w0 /tmp/<slug>-practice.zip   # paste into <script type="text/plain" id="nbZip">…</script>
```

Then point the `NB` map at it (`{enc: {zip: 'nbZip', file: '<slug>-practice.zip', open:
'<slug>-practice.ipynb'}}`) and use a `data-nb="enc"` button inside `.homework-final`. The handler
already tries `window.claude.use('downloads')` (Claude artifacts) and falls back to a normal blob
download. Tell the reader it's a zip, which apps open it (Jupyter, VS Code, Google Colab), and that it
needs only NumPy. Commit the `.ipynb` and the generator to `notes/` too.

---

## 9. Phase 7: the printable worksheet

The "Make your own worksheet" section generates a pen-and-paper version of a whole pass:

- Option chips (`#wsOpts`): which parts it covers, input length, widths, heads, whole vs. one-decimal
  numbers, answer key on/off. A live estimate of how many boxes there are to fill.
- `wsBuild(opts)` draws seeded numbers (`mulberry(seed * 7919 + 13)`) from small value sets so the
  sums stay pen-friendly; `wsHTML(D, key)` lays out each step with the given numbers, an empty box per
  answer, a hint under every step and the answer key at the back, labelled with the worksheet number.
- Print renders into `#printRoot` and toggles `html.ws-printing` so only the worksheet prints (A4,
  13mm margins); "Download to print later" saves a standalone HTML file.

Adapt the step list and box counter (`wsBoxes`) to the new architecture, cap the sizes so one pass stays
a few pages, print one and save it as `worksheets/sample-<part>-worksheet.pdf`.

---

## 10. Phase 8: add it to the hub

**Hub card**: in the root `index.html`, copy the existing card, number it, and keep the "Next up…" card
last with the next number:

```html
<li class="card">
  <span class="num">02</span>
  <h3>BERT, by hand</h3>
  <p class="paper"><a href="https://arxiv.org/abs/1810.04805">BERT: Pre-training of Deep Bidirectional Transformers</a> · Devlin et al., 2018</p>
  <p class="desc">One or two sentences: what the reader walks through, from input to output.</p>
  <ul class="tags"><li>encoder</li><li>masked LM</li><li>notebooks</li><li>worksheet</li></ul>
  <a class="go" href="bert/">Start →</a>
</li>
<li class="card soon">
  <span class="num">03</span>
  <h3>Next up…</h3>
  <p>More architectures are on the way. Star the repo to follow along.</p>
</li>
```

**README**: add a row to the Breakdowns table:

```markdown
| 02 | **BERT, by hand** | [*BERT: Pre-training of Deep Bidirectional Transformers*](https://arxiv.org/abs/1810.04805) (Devlin et al., 2018) | [Site](https://guptaaditya95.github.io/llmbreakdown/bert/) · [Folder](bert/) |
```

and add the folder to "Repo layout" if it has a new kind of file.

---

## 11. Phase 9: push to GitHub and publish

### 11.1 Today: `GuptaAditya95/llmbreakdown`

- Pages is set to **Deploy from a branch → `main` → `/ (root)`**, so anything merged to `main` goes live
  in a minute or two. No build step, no Jekyll config needed (plain HTML).
- Work on a branch, open a pull request, check it, merge:

```bash
git clone https://github.com/GuptaAditya95/llmbreakdown && cd llmbreakdown
git checkout -b add-<slug>
# add <slug>/index.html, <slug>/notes/, <slug>/worksheets/, edit index.html and README.md
git add -A && git commit -m "Add <Name>, by hand"
git push -u origin add-<slug>
# open a PR on GitHub, merge it into main
```

- After merging, check `https://guptaaditya95.github.io/llmbreakdown/<slug>/` and the hub card link.
  The repo's Actions tab shows the "pages build and deployment" run if it's slow to appear.
- When Claude does the push: the Claude GitHub App must be installed on the repo with access to
  `llmbreakdown` (already done), and Claude can't create new repositories on your account, so create any
  new repo yourself first.

### 11.2 Keep one repo, one folder per breakdown

Don't make a repo per architecture. Every breakdown is a folder in this repo so the hub, the shared look
and the relative links stay together.

### 11.3 Pending option: `letsbreakitdown.github.io`

The hub was meant to live at `letsbreakitdown.github.io`. That address needs a GitHub **organisation**
(or user) named `letsbreakitdown`. If you decide to do it:

1. Create the free organisation `letsbreakitdown` on GitHub (you, logged in: Your organizations → New).
2. Either transfer `llmbreakdown` into it and rename it to `letsbreakitdown.github.io` (Settings →
   Danger zone → Transfer, then Rename), or create `letsbreakitdown/letsbreakitdown.github.io` and push
   this repo's contents to it.
3. In the new repo's Settings → Pages, deploy from `main`, `/ (root)`.
4. Install the Claude GitHub App on the organisation so Claude can push there.
5. Update the absolute URLs: the README "Live site" line and table links, and the hub footer's star link.

Then the hub is at `https://letsbreakitdown.github.io/` and each breakdown at
`https://letsbreakitdown.github.io/<slug>/`. GitHub redirects the old repo URL after a transfer, but the
old `guptaaditya95.github.io/llmbreakdown/` Pages address will stop serving, so update anywhere you've
shared it. (If you keep the repo name `llmbreakdown` inside the org, the address becomes
`letsbreakitdown.github.io/llmbreakdown/` instead.)

---

## 12. QA checklist before merging

- [ ] One set of sizes from start to finish; every step shows run and paper shapes, and changing the
      settings updates all of them, the header and the shape bar.
- [ ] Every computing step has a practice panel; every answer checks out with "Show answer" and "New
      numbers" still produces solvable tasks.
- [ ] At 360px wide: no horizontal page scroll, matrices scroll inside their cards, numbers never touch,
      the practice sheet opens and closes and Escape closes it.
- [ ] Dark mode (system setting) looks right: no white boxes, readable red and blue.
- [ ] Reduced motion: no draw-on or scroll animation.
- [ ] Both notebook downloads unzip and run top to bottom; with the reference answers every check prints ✓.
- [ ] The worksheet generates, prints only the worksheet, and its answer key matches.
- [ ] Section references (§) match the arXiv version; the citation tabs are correct for the new paper.
- [ ] The cover kicker links to `../`; the hub card links to `<slug>/`; README row added.
- [ ] Credits in "Thank you", the "Made with Claude" footer and the stand-in-weights note are present.
- [ ] No console errors in the browser.

---

## 13. Prompt to hand to Claude

Paste this into a new thread, filling in the brackets:

> Build the next breakdown for my "Let's break it down" site, following `PLAYBOOK.md` in
> GuptaAditya95/llmbreakdown exactly, using `attention/index.html` as the template.
>
> - Architecture and paper: [name], [paper title, authors, year, arXiv link]
> - Folder/slug: [slug]
> - Parts: [e.g. one part: "The decoder"; or two parts: ...]
> - Recommended run sizes: [e.g. n = 6, d_model = 8, h = 2, d_ff = 32] (paper: [values])
> - The idea I want to linger on: [e.g. the [CLS] token / patch embedding / selective scan]
> - Thank-you credits: [people and links]
>
> Start with the notes (walkthrough with shapes, every calculation, a NumPy check), show them to me, then
> build the site, the per-step practice, the notebook(s) and the worksheet, add the hub card and README
> row, and open a PR. Run the QA checklist and tell me anything that doesn't pass.

Work in that order and review between phases: notes first (that's where your understanding is checked),
then the page, then the homework.
