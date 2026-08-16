# What changed (now fully wired up, no more open TODOs)

Now that I could see `utils/`, everything requested is actually connected
end-to-end — no more "you'll need to add this" gaps. 6 files changed:

**Pages** (`pages/`): `1_homepage.py`, `2_movie_details.py`, `4_prediction_history.py`
**Utils** (`utils/`): `cache_store.py`, `reddit_client.py`, `youtube_client.py`

`tmdb_client.py`, `model_loader.py`, `feature_engineering.py`, `navbar.py`,
`styling.py`, `movie_grid.py` are untouched — drop these six files in over
the matching ones and you're set.

## Homepage
- Custom hero background: drop an image at `assets/homepage_bg.jpg` (or
  `.png`) — it's read and base64-encoded automatically, layered under the
  purple gradient. No image → gradient-only fallback (the old code pointed
  at a webpage URL, not an image, so nothing rendered before).
- Trending row + search results: posters are now `<a href="/movie_details?movie_id=...">`
  directly — click the poster, no separate button/expander.
- Popular Trailers: thumbnails link to `/homepage?play_video=<id>`
  (same-page query param), which plays the video inline. No ▶ button row.

## Movie Details
- Poster fixed at `width=220` (was stretching full-column); metadata font
  bumped to 15px with more line-height.
- Three-column header: poster | metadata + button | **prediction result**.
  Cached titles show their result immediately, no click needed.
- YouTube panel now plays the fetched trailer via `st.video(...)`.
- Reddit sentiment renders as a custom red→green **hue bar** instead of
  `st.progress`.
- Social panels persist in `st.session_state` once fetched live, so they
  stay visible after a prediction is cached. For a title cached in an
  *earlier* session (nothing in memory), there's a "🔄 Refresh social
  signals" button instead of a dead caption.
- New "📝 Sample sentiment fetched" expander — shows the 3 most
  sentiment-extreme Reddit posts and YouTube comments, each with a
  hue-coloured compound score. This needed `reddit_client.py` and
  `youtube_client.py` to return a `samples` list, which I added to both
  (`_compute_sentiment` now tracks `(text, compound)` pairs and returns the
  top 3 by `|compound|` — same aggregation logic, no new API calls).
- Accepts `?movie_id=` in the URL as a fallback to `session_state`, so
  poster links from the homepage work directly.

## Prediction History
- New "Manage predictions" section: multiselect titles → confirmation
  checkbox → "🗑 Delete selected".
- Added `cache_store.delete_movie(titles)` — case-insensitive match against
  `Title`, rewrites the CSV, returns the count deleted. The page calls this
  directly rather than touching the CSV itself.
