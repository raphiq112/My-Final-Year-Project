"""
Sentiment Word Cloud Helper
=============================
Builds a word cloud image from the raw Reddit posts / YouTube comments
fetched for a movie, so users get a qualitative feel for what people are
actually saying — as a complement to (not an input for) the VADER
aggregate scores. Purely descriptive/exploratory: nothing here touches
the feature vectors the models were trained on.

Suggested by the project supervisor as a way to visually corroborate the
sentiment numbers.
"""

from collections import Counter
import re

try:
    from wordcloud import WordCloud, STOPWORDS
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Generic terms that show up constantly in movie discussion/trailer comments
# but carry no sentiment signal of their own — without excluding these,
# they tend to dominate the cloud and crowd out anything informative.
GENERIC_STOPWORDS = {
    "movie", "movies", "film", "films", "trailer", "trailers", "official",
    "watch", "watching", "watched", "see", "seeing", "saw", "will", "just",
    "really", "trailer1", "http", "https", "www", "com", "com watch",
    "www youtube", "youtube", "amp", "gt", "one", "going", "make", "made",
    "http www",
}


def _title_stopwords(movie_title: str) -> set:
    """Movie-title words are the single biggest source of dead weight in a
    per-movie cloud (e.g. every 'Enola Holmes 3' post says 'Enola Holmes 3'),
    so they're stripped out along with the generic list above."""
    return {w.lower() for w in re.findall(r"[a-zA-Z']+", movie_title or "")}


def build_wordcloud_image(texts, movie_title: str = "", width=600, height=320):
    """
    Returns a PIL Image of the word cloud, or None if there isn't enough
    text to build one (or the `wordcloud` package isn't installed).
    """
    if not WORDCLOUD_AVAILABLE:
        return None

    joined = " ".join(t for t in texts if t and t.strip())
    if len(joined.split()) < 5:
        return None

    stopwords = set(STOPWORDS) | GENERIC_STOPWORDS | _title_stopwords(movie_title)

    wc = WordCloud(
        width=width, height=height, background_color=None, mode="RGBA",
        stopwords=stopwords, colormap="viridis", max_words=80,
        collocations=False, prefer_horizontal=0.9,
    ).generate(joined)

    return wc.to_image()


def top_words(texts, movie_title: str = "", n=15):
    """Fallback for when the `wordcloud` package isn't available — a plain
    frequency count of the same filtered vocabulary, still useful on its
    own as a quick 'what are people saying' signal."""
    stopwords = set(STOPWORDS if WORDCLOUD_AVAILABLE else []) | GENERIC_STOPWORDS | _title_stopwords(movie_title)
    words = []
    for t in texts:
        if not t:
            continue
        words.extend(w.lower() for w in re.findall(r"[a-zA-Z']{3,}", t))
    words = [w for w in words if w not in stopwords]
    return Counter(words).most_common(n)
