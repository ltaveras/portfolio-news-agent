import os, json, hashlib
from datetime import datetime, timedelta, timezone

import requests
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from openai import OpenAI

TICKERS_DEFAULT = "UVIX,UUUU,URNJ,REMX,NLR,UFO,SMR,NUKZ,OKLO,ARKVX,INNOX"

def env(name, default=None, required=False):
    v = os.getenv(name, default)
    if required and (v is None or str(v).strip() == ""):
        raise RuntimeError(f"Missing env var: {name}")
    return v

def ascii_safe(s: str) -> str:
    """
    SendGrid (or underlying libs) can sometimes choke on certain unicode characters.
    This makes the email safe by converting to basic ASCII.
    """
    if s is None:
        return ""
    # Normalize a few common punctuation marks first
    s = (str(s)
         .replace("—", "-")
         .replace("–", "-")
         .replace("“", '"')
         .replace("”", '"')
         .replace("’", "'")
         .replace("\u00a0", " "))
    # Drop any remaining non-ascii
    return s.encode("ascii", errors="ignore").decode("ascii")

def finnhub_news(api_key, symbol, lookback_hours=24):
    to_dt = datetime.now(timezone.utc)
    from_dt = to_dt - timedelta(hours=lookback_hours)

    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": symbol,
        "from": from_dt.date().isoformat(),
        "to": to_dt.date().isoformat(),
        "token": api_key
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    items = r.json()

    filtered = []
    for it in items:
        dt = datetime.fromtimestamp(it.get("datetime", 0), tz=timezone.utc)
        if dt >= from_dt:
            filtered.append({
                "symbol": symbol,
                "headline": it.get("headline", ""),
                "summary": it.get("summary", ""),
                "source": it.get("source", ""),
                "url": it.get("url", ""),
                "datetime": dt.isoformat(),
                "category": it.get("category", "")
            })
    return filtered

def load_seen(path="seen.json"):
    try:
        with open(path, "r") as f:
            return set(json.load(f))
    except Exception:
        return set()

def save_seen(seen, path="seen.json"):
    with open(path, "w") as f:
        json.dump(sorted(list(seen)), f)

def item_id(item):
    raw = (
        item.get("symbol","") + "|" +
        item.get("headline","") + "|" +
        item.get("datetime","") + "|" +
        item.get("url","")
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def send_email(sendgrid_key, from_email, to_email, subject, content):
    sg = SendGridAPIClient(sendgrid_key)

    # Make the email robust against unicode encoding issues
    subject = ascii_safe(subject)
    content = ascii_safe(content)

    message = Mail(
        from_email=from_email,
        to_emails=to_email,
        subject=subject,
        plain_text_content=content
    )
    sg.send(message)

def llm_analyze(openai_key, model, tickers, news_items, mode="daily"):
    client = OpenAI(api_key=openai_key)

    # Keep payload reasonable
    compact = news_items[:80]

    system = (
        "You are a portfolio intelligence analyst. You are advice-only. "
        "You monitor the user's tickers and produce actionable, non-hyped analysis. "
        "You think outside the box: second-order effects, correlations, regulation, supply chain, dilution risk, "
        "short interest dynamics, ETF structure risks, and macro sensitivity. "
        "Be explicit about uncertainty and what would change your view. "
        "Do not fabricate events; use only the provided news items."
    )

    if mode == "daily":
        user_obj = {
            "task": "Create a daily briefing for the user's tickers based ONLY on the news_items provided.",
            "requirements": [
                "Group by ticker; include 0–3 bullets each; skip tickers with no meaningful items.",
                "Add an 'Outside-the-box insights' section connecting tickers/themes.",
                "Add 'Recommendations (advice-only)' per ticker: Hold/Add/Trim/Avoid + 1–2 sentence rationale.",
                "Add a 'Watch next' checklist (earnings, filings, catalysts).",
                "Keep it concise and skimmable."
            ],
            "tickers": tickers,
            "news_items": compact
        }
    else:
        user_obj = {
            "task": "Decide whether this is important enough to alert the user now (advice-only). Use ONLY the news_items provided.",
            "requirements": [
                "Score severity 0-100 (100 = immediate action).",
                "If severity < 70, say 'NO ALERT' and give a 2 sentence rationale.",
                "If severity >= 70, say 'ALERT' and provide: what happened, why it matters, what to do next (advice-only)."
            ],
            "tickers": tickers,
            "news_items": compact
        }

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":json.dumps(user_obj)}
        ],
        temperature=0.4
    )
    return resp.choices[0].message.content

def run_daily():
    finnhub_key = env("FINNHUB_API_KEY", required=True)
    openai_key = env("OPENAI_API_KEY", required=True)
    openai_model = env("OPENAI_MODEL", "gpt-4o-mini")
    sendgrid_key = env("SENDGRID_API_KEY", required=True)
    from_email = env("FROM_EMAIL", required=True)
    to_email = env("TO_EMAIL", required=True)

    tickers = [t.strip().upper() for t in env("TICKERS", TICKERS_DEFAULT).split(",") if t.strip()]
    lookback_hours = int(env("LOOKBACK_HOURS_DAILY", "24"))

    all_items = []
    for t in tickers:
        try:
            all_items.extend(finnhub_news(finnhub_key, t, lookback_hours=lookback_hours))
        except Exception as e:
            all_items.append({
                "symbol": t,
                "headline": f"[Data error fetching news for {t}]",
                "summary": str(e),
                "source": "system",
                "url": "",
                "datetime": datetime.now(timezone.utc).isoformat(),
                "category": "error"
            })

    all_items.sort(key=lambda x: x.get("datetime",""), reverse=True)

    content = llm_analyze(openai_key, openai_model, tickers, all_items, mode="daily")
    subject = f"Daily Portfolio Briefing - {datetime.now().strftime('%Y-%m-%d')}"
    send_email(sendgrid_key, from_email, to_email, subject, content)

def run_breaking():
    finnhub_key = env("FINNHUB_API_KEY", required=True)
    openai_key = env("OPENAI_API_KEY", required=True)
    openai_model = env("OPENAI_MODEL", "gpt-4o-mini")
    sendgrid_key = env("SENDGRID_API_KEY", required=True)
    from_email = env("FROM_EMAIL", required=True)
    to_email = env("TO_EMAIL", required=True)

    tickers = [t.strip().upper() for t in env("TICKERS", TICKERS_DEFAULT).split(",") if t.strip()]
    lookback_hours = int(env("LOOKBACK_HOURS_BREAKING", "6"))

    seen = load_seen("seen.json")

    new_items = []
    for t in tickers:
        try:
            items = finnhub_news(finnhub_key, t, lookback_hours=lookback_hours)
            for it in items:
                iid = item_id(it)
                if iid not in seen:
                    new_items.append(it)
                    seen.add(iid)
        except Exception:
            continue

    save_seen(seen, "seen.json")

    if not new_items:
        return

    new_items.sort(key=lambda x: x.get("datetime",""), reverse=True)
    content = llm_analyze(openai_key, openai_model, tickers, new_items, mode="breaking")

    # If LLM says no alert, do nothing.
    if "NO ALERT" in content and "ALERT" not in content:
        return

    subject = f"Portfolio Alert - {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
    send_email(sendgrid_key, from_email, to_email, subject, content)

if __name__ == "__main__":
    mode = env("MODE", "daily").lower().strip()
    if mode == "daily":
        run_daily()
    elif mode == "breaking":
        run_breaking()
    else:
        raise RuntimeError("MODE must be 'daily' or 'breaking'")
