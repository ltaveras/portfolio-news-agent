import os, json, time, hashlib
from datetime import datetime, timedelta, timezone

import requests
from dateutil import parser as dateparser
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from openai import OpenAI

TICKERS_DEFAULT = "UVIX,UUUU,URNJ,REMX,NLR,UFO,SMR,NUKZ,OKLO,ARKVX,INNOX"

def env(name, default=None, required=False):
    v = os.getenv(name, default)
    if required and not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v

def finnhub_news(api_key, symbol, lookback_hours=24):
    # Finnhub company-news endpoint (works best for equities; ETFs sometimes have sparse news)
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
    # Finnhub returns more than 24h sometimes due to date granularity; we filter precisely.
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
    raw = (item.get("symbol","") + "|" + item.get("headline","") + "|" + item.get("datetime","") + "|" + item.get("url",""))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def send_email(sendgrid_key, from_email, to_email, subject, content):
    sg = SendGridAPIClient(sendgrid_key)
    message = Mail(
        from_email=from_email,
        to_emails=to_email,
        subject=subject,
        plain_text_content=content
    )
    sg.send(message)

def llm_analyze(openai_key, model, tickers, news_items, mode="daily"):
    """
    mode = "daily" or "breaking"
   
    """
    client = OpenAI(api_key=openai_key)

    # Keep payload compact
    compact = news_items[:80]

    system = (
        "You are a portfolio intelligence analyst. You are advice-only. "
        "You monitor the user's tickers and produce actionable, non-hyped analysis. "
        "You think 'outside the box': second-order effects, correlations, regulation, supply chain, dilution risk, "
        "short interest dynamics, ETF structure risks, and macro sensitivity. "
        "You must be explicit about uncertainty and what would change your view."
    )

    if mode == "daily":
        user = {
            "task": "Create a daily briefing for the user's tickers.",
            "requirements": [
                "Use only the provided news items; do not fabricate events.",
                "Group by ticker; include 0–3 bullets each (skip tickers with no meaningful items).",
                "Add an 'Outside-the-box insights' section connecting tickers/themes.",
                "Add 'Recommendations (advice-only)' per ticker: Hold/Add/Trim/Avoid + 1–2 sentence rationale.",
                "Add a 'Watch next' checklist (earnings, filings, catalysts to monitor).",
                "Keep it concise, skimmable, and clear."
            ],
            "tickers": tickers,
            "news_items": compact
        }
    else:
        user = {
            "task": "Decide whether this is important enough to alert the user now.",
            "requirements": [
                "Score severity 0-100 (100 = immediate action).",
                "If severity < 70, say 'NO ALERT' and give a 2 sentence rationale.",
                "If severity >= 70, say 'ALERT' and provide: what happened, why it matters, what to do next (advice-only).",
                "Do not fabricate; use only provided news items."
            ],
            "tickers": tickers,
            "news_items": compact
        }

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":json.dumps(user)}
        ],
        temperature=0.4
    )
    text = resp.choices[0].message.content

  
    return text

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

    # Sort by time desc
    all_items.sort(key=lambda x: x.get("datetime",""), reverse=True)

    html = llm_analyze(openai_key, openai_model, tickers, all_items, mode="daily")
    subject = f"Daily Portfolio Briefing - {datetime.now().strftime('%Y-%m-%d')}"

    send_email(sendgrid_key, from_email, to_email, subject, html)

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

    # Analyze only new items
    new_items.sort(key=lambda x: x.get("datetime",""), reverse=True)
    html = llm_analyze(openai_key, openai_model, tickers, new_items, mode="breaking")

    # If LLM says "NO ALERT" we skip emailing; otherwise send
    if "NO ALERT" in html and "ALERT" not in html:
        return

    subject = f"Daily Portfolio Briefing - {datetime.now().strftime('%Y-%m-%d')}"
    send_email(sendgrid_key, from_email, to_email, subject, html)

if __name__ == "__main__":
    mode = env("MODE", "daily").lower().strip()
    if mode == "daily":
        run_daily()
    elif mode == "breaking":
        run_breaking()
    else:
        raise RuntimeError("MODE must be 'daily' or 'breaking'")
