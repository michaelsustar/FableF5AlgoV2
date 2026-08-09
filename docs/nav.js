/* DocShuuBets shared header.
   ------------------------------------------------------------------
   TO ADD A NEW ALGORITHM: add one entry to the sport's `algos` array
   below. Every page picks it up automatically — no other edits.
   `href` is relative to the site root (docs/).

   TO RETIRE ONE: move its entry into the Archived list and repoint
   `href` at archive/. The model keeps running and grading — this file
   only controls where the site links to it.
   ------------------------------------------------------------------ */
const SPORTS = [
  {
    name: "MLB",
    algos: [
      {
        name: "First Five Totals",
        note: "Live · F5 over/under",
        href: "mlb/first-five-totals.html"
      }
    ]
  },
  { name: "NBA", algos: [] },
  { name: "NFL", algos: [] },
  {
    name: "Archived",
    algos: [
      {
        name: "First Five Hot Streak",
        note: "Off the board · still graded",
        href: "archive/first-five.html"
      }
    ]
  }
];

(function () {
  const root = window.SITE_ROOT || "";      // "../" on pages in subfolders

  const menu = s => s.algos.length
    ? s.algos.map(a =>
        `<a href="${root}${a.href}">${a.name}<small>${a.note || ""}</small></a>`
      ).join("")
    : `<span class="soon">In development<small>nothing published yet</small></span>`;

  const html = `
    <div class="wrap">
      <a class="brand" href="${root}index.html">Doc<em>Shuu</em>Bets</a>
      <nav class="nav" aria-label="Sports">
        ${SPORTS.map((s, i) => `
          <div class="item" data-i="${i}">
            <button type="button" aria-expanded="false" aria-haspopup="true">
              ${s.name}<span class="caret">▼</span>
            </button>
            <div class="menu" role="menu">${menu(s)}</div>
          </div>`).join("")}
      </nav>
    </div>`;

  const bar = document.createElement("header");
  bar.className = "masthead";
  bar.innerHTML = html;
  document.body.prepend(bar);

  const items = [...bar.querySelectorAll(".item")];
  const closeAll = except => items.forEach(it => {
    if (it !== except) {
      it.classList.remove("open");
      it.querySelector("button").setAttribute("aria-expanded", "false");
    }
  });

  items.forEach(it => {
    const btn = it.querySelector("button");
    btn.addEventListener("click", e => {
      e.stopPropagation();
      const open = it.classList.toggle("open");
      btn.setAttribute("aria-expanded", open ? "true" : "false");
      closeAll(it);
    });
    // hover-to-open on real pointers; taps still work via the click handler
    if (window.matchMedia("(hover:hover)").matches) {
      it.addEventListener("mouseenter", () => {
        closeAll(it);
        it.classList.add("open");
        btn.setAttribute("aria-expanded", "true");
      });
      it.addEventListener("mouseleave", () => {
        it.classList.remove("open");
        btn.setAttribute("aria-expanded", "false");
      });
    }
  });

  document.addEventListener("click", () => closeAll(null));
  document.addEventListener("keydown", e => { if (e.key === "Escape") closeAll(null); });
})();
