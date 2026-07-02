# Deploying the CERT-FLOW project page (GitHub Pages)

Run when the visualization suite is verified and its MP4s + posters are in
`site/assets/media/` and `site/assets/img/` (the page references them).

## One-time: publish to a gh-pages branch and enable Pages

```bash
# from repo root, on a clean tree, as the Archerkattri account
gh auth switch --user Archerkattri

# build an orphan gh-pages branch containing ONLY the site/ contents at root.
# NOTE: `git switch --orphan` EMPTIES the working tree on current git, so stage
# the site contents OUTSIDE the repo first, then copy them back in.
STAGE=$(mktemp -d) && cp -r site/* "$STAGE"/ && cp -r assets "$STAGE"/assets
git switch --orphan gh-pages
git rm -rq --cached . 2>/dev/null || true
cp -r "$STAGE"/* . && rm -rf "$STAGE"
# the source page references ../assets/; at gh-pages root it must be assets/
sed -i 's#\.\./assets/#assets/#g' index.html
touch .nojekyll           # serve assets verbatim (no Jekyll processing)
git add index.html assets .nojekyll DEPLOY.md 2>/dev/null
GIT_AUTHOR_NAME="Krishi Attri" GIT_AUTHOR_EMAIL="krishiattriwork@gmail.com" \
GIT_COMMITTER_NAME="Krishi Attri" GIT_COMMITTER_EMAIL="krishiattriwork@gmail.com" \
  git commit -q -m "Project page"
git push -q -f origin gh-pages
git switch main

# enable Pages from the gh-pages branch root (one time)
gh api -X POST repos/Archerkattri/CERT-FLOW/pages \
  -f 'source[branch]=gh-pages' -f 'source[path]=/' 2>/dev/null || \
gh api -X PUT repos/Archerkattri/CERT-FLOW/pages \
  -f 'source[branch]=gh-pages' -f 'source[path]=/'
```

Live URL: https://archerkattri.github.io/CERT-FLOW/
(Updates: re-run the orphan-branch build + push; Pages redeploys automatically.)

NOTE: the `site/` source also lives on `main` for version control; only the
built copy goes to `gh-pages`. Keep media files small (<6 MB each).
