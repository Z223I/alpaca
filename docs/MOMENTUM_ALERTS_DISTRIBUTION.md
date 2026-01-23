# Momentum Alerts Distribution Strategy

This document discusses options for displaying and distributing stock momentum alerts beyond Telegram, including web/mobile platforms and cloud storage considerations.

## Current State

- **Data Generation**: Alerts generated on home server
- **Storage**: Local JSON files organized by date (`historical_data/YYYY-MM-DD/momentum_alerts_sent/bullish/`)
- **Distribution**: Telegram messages
- **Data Volume**: ~37 trading days of historical data, multiple alerts per day

## Data Structure Summary

Each alert contains rich trading data:
- **Price Data**: Current price, market open, VWAP, EMA indicators
- **Momentum Metrics**: Short/long momentum, squeeze detection
- **Volume Data**: Current volume, surge detection, float rotation
- **Fundamentals**: Market cap, shares outstanding, float shares
- **Metadata**: Timestamp, urgency level, recipients, sources

---

## Part 1: Display Options

### Option A: Progressive Web App (PWA)

**Recommended for starting out.**

A PWA works on both desktop and mobile browsers, can be "installed" on phones, and supports push notifications.

**Pros:**
- Single codebase for web and mobile
- No app store approval process
- Push notifications via Web Push API
- Works offline with service workers
- Lower development cost than native apps

**Cons:**
- iOS Safari has some PWA limitations (no background sync)
- Less "native" feel than true mobile apps
- Push notifications require user permission

**Technology Stack:**
```
Frontend: React/Vue/Svelte + TailwindCSS
Backend API: FastAPI (Python) or Node.js
Real-time: WebSockets or Server-Sent Events (SSE)
Hosting: Vercel, Netlify, or Cloudflare Pages
```

### Option B: Native Mobile Apps

**Consider after validating market demand.**

**Pros:**
- Best user experience
- Reliable push notifications
- App store presence (discovery/credibility)
- Access to device features

**Cons:**
- Two codebases (iOS + Android) or cross-platform framework
- App store review process and fees ($99/year Apple, $25 one-time Google)
- Higher development and maintenance cost

**Technology Options:**
- **Cross-platform**: React Native, Flutter
- **Native**: Swift (iOS), Kotlin (Android)

### Option C: Hybrid Approach

Start with a PWA, add native apps later for subscribers who want the best experience.

---

## Part 2: Cloud Storage Architecture

### Why Move to the Cloud?

| Concern | Home Server | Cloud |
|---------|-------------|-------|
| Reliability | Single point of failure | High availability |
| Latency | Depends on home internet | Global CDN |
| Scalability | Limited | Elastic |
| Security | You manage everything | Managed infrastructure |
| Cost | Electricity + hardware | Pay-per-use |

### Recommended Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        HOME SERVER                               │
│  ┌─────────────────┐                                            │
│  │ Alert Generator │──────┐                                     │
│  └─────────────────┘      │                                     │
└───────────────────────────│─────────────────────────────────────┘
                            │
                            ▼ Push alerts via HTTPS
┌─────────────────────────────────────────────────────────────────┐
│                         CLOUD                                    │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │   API/Edge   │───▶│   Database   │───▶│   File Storage   │  │
│  │  (FastAPI)   │    │ (PostgreSQL) │    │ (S3/R2 for JSON) │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│         │                    │                                   │
│         │                    ▼                                   │
│         │           ┌──────────────┐                            │
│         │           │    Redis     │ (optional: caching,        │
│         │           │              │  pub/sub for real-time)    │
│         │           └──────────────┘                            │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐    ┌──────────────┐                           │
│  │   Web App    │    │ Push Service │                           │
│  │    (PWA)     │    │  (FCM/APNS)  │                           │
│  └──────────────┘    └──────────────┘                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Cloud Provider Comparison

#### Option 1: Cloudflare (Recommended for Starting)

**Why**: Generous free tier, global edge network, simple pricing.

| Service | Purpose | Free Tier |
|---------|---------|-----------|
| Workers | API/serverless functions | 100K requests/day |
| R2 | Object storage (S3-compatible) | 10GB storage, no egress fees |
| D1 | SQLite database at edge | 5GB storage |
| Pages | Static site hosting | Unlimited |

**Estimated Cost at Scale**: $5-20/month for moderate traffic

#### Option 2: AWS

**Why**: Most comprehensive, best for complex needs.

| Service | Purpose | Notes |
|---------|---------|-------|
| Lambda | Serverless API | Pay per invocation |
| DynamoDB | NoSQL database | Good for time-series alert data |
| S3 | JSON file storage | Cheap storage, egress fees apply |
| CloudFront | CDN | Global distribution |
| SNS/SQS | Push notifications | Reliable messaging |

**Estimated Cost**: $10-50/month depending on usage

#### Option 3: Supabase (PostgreSQL + Realtime)

**Why**: Firebase alternative with PostgreSQL, great for real-time.

| Feature | Benefit |
|---------|---------|
| PostgreSQL | Full SQL, JSON support |
| Realtime | Built-in WebSocket subscriptions |
| Auth | User management included |
| Storage | File storage for JSON/images |

**Free Tier**: 500MB database, 1GB storage, 2GB bandwidth
**Estimated Cost**: $25/month Pro tier

### Database Schema Recommendation

For alert data, a hybrid approach works well:

```sql
-- Main alerts table (PostgreSQL)
CREATE TABLE alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(10) NOT NULL,
    alert_type VARCHAR(20) NOT NULL,  -- 'bullish', 'bearish'
    urgency VARCHAR(20),               -- 'urgent', 'normal'

    -- Price data
    current_price DECIMAL(10,4),
    market_open_price DECIMAL(10,4),
    percent_gain DECIMAL(8,4),
    vwap DECIMAL(10,6),

    -- Momentum
    momentum DECIMAL(8,6),
    momentum_short DECIMAL(8,6),

    -- Metadata
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Store full JSON for flexibility
    raw_data JSONB NOT NULL,

    -- Indexes
    INDEX idx_symbol (symbol),
    INDEX idx_timestamp (timestamp DESC),
    INDEX idx_urgency (urgency)
);
```

**Rationale**:
- Indexed columns for fast queries (symbol, date, urgency)
- JSONB column preserves full alert data for flexibility
- Can query specific fields OR full JSON as needed

---

## Part 3: Real-Time vs Batch Updates

### For Trading Alerts: Real-Time is Critical

Users need alerts immediately. Options:

1. **WebSockets**: Best for persistent connections
   - Use Socket.IO or native WebSockets
   - Server pushes alerts instantly

2. **Server-Sent Events (SSE)**: Simpler than WebSockets
   - One-way server-to-client
   - Works through proxies/firewalls better
   - Good browser support

3. **Push Notifications**: For mobile/background
   - Firebase Cloud Messaging (FCM) for Android/Web
   - Apple Push Notification Service (APNS) for iOS

### Recommended Flow

```
Home Server generates alert
         │
         ▼
POST to Cloud API (HTTPS)
         │
         ▼
Cloud API:
  1. Store in database
  2. Publish to Redis pub/sub
  3. Trigger push notification
         │
         ├──▶ WebSocket broadcast to connected clients
         │
         └──▶ FCM/APNS push to mobile devices
```

---

## Part 4: Monetization Considerations

### Subscription Tiers

| Tier | Features | Price Point |
|------|----------|-------------|
| Free | Delayed alerts (15-30 min), limited history | $0 |
| Basic | Real-time alerts, 7-day history | $19-29/month |
| Pro | Real-time + all indicators, full history, API access | $49-99/month |

### Technical Implementation

```python
# Example: Tier-based alert filtering
class AlertDistributor:
    def distribute(self, alert: dict, user: User):
        if user.tier == 'free':
            # Delay alert by 15 minutes
            schedule_delayed_send(alert, delay_minutes=15)
        elif user.tier == 'basic':
            # Send immediately, but limited fields
            send_basic_alert(alert, user)
        else:
            # Full alert with all indicators
            send_full_alert(alert, user)
```

### Payment Methods

#### Option 1: Stripe (Recommended)

**Why**: Industry standard, excellent documentation, handles subscriptions natively.

| Feature | Details |
| ------- | ------- |
| Subscription billing | Built-in recurring payments, trials, upgrades/downgrades |
| Payment methods | Credit/debit cards, Apple Pay, Google Pay, bank transfers |
| Fees | 2.9% + $0.30 per transaction (US) |
| Payout schedule | 2-day rolling payouts |
| Tax handling | Stripe Tax add-on for automatic sales tax |

**Integration Example:**

```python
import stripe

stripe.api_key = os.environ["STRIPE_SECRET_KEY"]

# Create a subscription checkout session
@app.post("/api/create-checkout")
async def create_checkout(tier: str, user_id: str):
    price_ids = {
        "basic": "price_basic_monthly_id",
        "pro": "price_pro_monthly_id"
    }

    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": price_ids[tier], "quantity": 1}],
        success_url="https://yourapp.com/success?session_id={CHECKOUT_SESSION_ID}",
        cancel_url="https://yourapp.com/pricing",
        metadata={"user_id": user_id}
    )
    return {"checkout_url": session.url}
```

#### Option 2: Paddle

**Why**: Merchant of record (handles taxes/VAT globally for you).

| Feature | Details |
| ------- | ------- |
| Tax compliance | Paddle handles all sales tax/VAT worldwide |
| Subscription billing | Full subscription management |
| Fees | 5% + $0.50 per transaction |
| Benefit | No need to register for tax in multiple jurisdictions |

**Best for**: Selling internationally without tax compliance headaches.

#### Option 3: LemonSqueezy

**Why**: Built for digital products, simpler than Stripe.

| Feature | Details |
| ------- | ------- |
| Merchant of record | Handles taxes globally |
| Fees | 5% + $0.50 per transaction |
| Focus | Digital products and SaaS |
| Affiliate system | Built-in affiliate program |

#### Option 4: PayPal (Additional Option)

**Why**: Some users prefer PayPal; offer as secondary option.

| Feature | Details |
| ------- | ------- |
| Subscriptions | PayPal Subscriptions API |
| Fees | 2.9% + $0.30 per transaction |
| User trust | High brand recognition |
| Limitation | Less developer-friendly than Stripe |

#### Payment Recommendation

| Situation | Recommendation |
| --------- | -------------- |
| US-only customers | Stripe |
| International customers | Paddle or LemonSqueezy (tax handling) |
| Maximum payment options | Stripe + PayPal as backup |
| Simplest setup | LemonSqueezy |

### Crowdfunding with GoFundMe

Use crowdfunding to fund initial development costs before launching subscriptions.

#### Why GoFundMe

| Aspect | Details |
| ------ | ------- |
| Platform fee | 0% platform fee (GoFundMe takes nothing) |
| Payment processing | 2.9% + $0.30 per donation |
| Payout speed | Funds available within 2-5 business days |
| Trust factor | Well-known platform, donors feel secure |
| Flexibility | No deadlines, keep what you raise |

#### Campaign Strategy

**Funding Goal Breakdown:**

| Item | Estimated Cost |
| ---- | -------------- |
| Cloud infrastructure (1 year) | $200 - $500 |
| Domain and SSL | $50 - $100 |
| Development tools/services | $100 - $300 |
| Marketing budget | $200 - $500 |
| Legal (Terms of Service, Privacy Policy) | $200 - $500 |
| Contingency buffer | $200 - $500 |
| **Total** | **$950 - $2,400** |

**Campaign Page Elements:**

1. **Compelling story**: Why you built the alerts, your trading background
2. **Demo video**: Show the alerts in action (screen recording from your YouTube)
3. **Transparency**: Breakdown of how funds will be used
4. **Reward tiers**: Offer early supporter benefits

#### Reward Tiers for Backers

| Donation | Reward |
| -------- | ------ |
| $10+ | Name on "Founding Supporters" page |
| $25+ | 1 month free Pro access at launch |
| $50+ | 3 months free Pro access at launch |
| $100+ | 6 months free Pro access + Discord role |
| $250+ | Lifetime Basic tier + founding member badge |
| $500+ | Lifetime Pro tier + 1-on-1 onboarding call |

#### Campaign Promotion via YouTube

```
┌─────────────────────────────────────────────────────────┐
│                    YouTube Video                         │
│  "I'm Building a Stock Alert Service - Here's How       │
│   You Can Help (And Get Early Access)"                  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  Video Description                       │
│                                                          │
│  Support the project: https://gofundme.com/your-link    │
│                                                          │
│  Rewards:                                                │
│  - $25 = 1 month free Pro                               │
│  - $50 = 3 months free Pro                              │
│  - $100 = 6 months free + Discord role                  │
└─────────────────────────────────────────────────────────┘
```

**Video Content Ideas:**

- Behind-the-scenes of alert system development
- Live demo showing alerts catching big movers
- Explanation of your momentum algorithm
- Q&A addressing viewer questions about the service

#### GoFundMe Alternatives

| Platform | Fee | Best For |
| -------- | --- | -------- |
| **GoFundMe** | 2.9% + $0.30 | General crowdfunding, no platform fee |
| **Kickstarter** | 5% + 3-5% processing | Product launches, all-or-nothing model |
| **Indiegogo** | 5% + 3-5% processing | Flexible funding, tech products |
| **Buy Me a Coffee** | 5% | Ongoing small donations, creator-focused |
| **Patreon** | 5-12% | Recurring support, content creators |
| **Ko-fi** | 0% (optional tip) | One-time donations, simple setup |

#### Recommended Approach

**Phase 1: Pre-Launch Funding**

1. Create GoFundMe campaign with clear goals
2. Announce on YouTube with dedicated video
3. Share in video descriptions and community posts
4. Offer early access rewards to incentivize

**Phase 2: Ongoing Support (Post-Launch)**

1. Add "Buy Me a Coffee" or Ko-fi for tips
2. Include in app footer: "Support development"
3. Mention in YouTube videos occasionally

**Integration Example (Landing Page):**

```html
<!-- Support section on your website -->
<section class="support-development">
  <h2>Support Development</h2>
  <p>Help us build the best momentum alert service.</p>

  <div class="support-options">
    <!-- GoFundMe for initial funding -->
    <a href="https://gofundme.com/your-campaign" class="btn-gofundme">
      Back the Project on GoFundMe
    </a>

    <!-- Ko-fi for ongoing tips -->
    <a href="https://ko-fi.com/yourusername" class="btn-kofi">
      Buy Me a Coffee
    </a>
  </div>

  <div class="founding-supporters">
    <h3>Founding Supporters</h3>
    <ul>
      <!-- Dynamically populated from database -->
      <li>John D. - $100</li>
      <li>Sarah M. - $50</li>
      <!-- ... -->
    </ul>
  </div>
</section>
```

#### Tax Considerations

| Situation | Tax Treatment |
| --------- | ------------- |
| Pure donations (no reward) | May be taxable income |
| Donations with rewards | Taxable as sales revenue |
| Significant amounts | Consult a tax professional |

**Note**: GoFundMe donations are generally considered taxable income in the US. Keep records of all donations and associated rewards for tax purposes.

### Marketing via YouTube

Your YouTube channel is a strong asset for customer acquisition.

#### Content Strategy

| Content Type | Purpose | Call to Action |
| ------------ | ------- | -------------- |
| Live trading sessions | Show alerts in real-time | "Get these alerts live at..." |
| Daily recap videos | Review which alerts hit | "Join to get tomorrow's alerts" |
| Educational content | Explain momentum indicators | Build trust, soft sell |
| Alert breakdowns | Deep-dive on winning trades | Demonstrate value |

#### YouTube-to-Subscriber Funnel

```
YouTube Video
     │
     ▼
Video Description: Link to landing page
     │
     ▼
Landing Page: Free tier signup (email capture)
     │
     ▼
Email Nurture: Show value, delayed alerts
     │
     ▼
Upgrade CTA: "Get real-time alerts"
     │
     ▼
Paid Subscription via Stripe/Paddle
```

#### Integration Ideas

1. **Live Alert Overlay**: Show alerts on-screen during live streams
2. **Discord/Community**: Free Discord for engagement, premium channel for subscribers
3. **Affiliate Program**: Let viewers earn commission for referrals (LemonSqueezy has this built-in)
4. **YouTube Memberships**: Alternative/supplement to direct subscriptions

#### Tracking Attribution

```python
# Track which YouTube video drove the signup
@app.post("/api/signup")
async def signup(email: str, utm_source: str = None, utm_campaign: str = None):
    user = await create_user(email)

    # Store attribution for analytics
    if utm_source == "youtube":
        await store_attribution(user.id, {
            "source": "youtube",
            "campaign": utm_campaign,  # e.g., "live-trading-jan-2026"
            "signup_date": datetime.now()
        })

    return {"success": True}
```

Use UTM parameters in your YouTube links:
`https://yourapp.com/signup?utm_source=youtube&utm_campaign=video-name`

---

## Part 5: Implementation Roadmap

### Phase 1: Cloud Foundation (Week 1-2)

1. Set up Cloudflare account (or chosen provider)
2. Create API endpoint to receive alerts from home server
3. Store alerts in D1/PostgreSQL
4. Basic web dashboard to view alerts

**Home Server Change**: Add HTTP POST to cloud API after each alert.

### Phase 2: Web App (Week 3-4)

1. Build PWA with React/Vue
2. Real-time updates via SSE or WebSockets
3. Alert filtering (by symbol, urgency, time)
4. User authentication (Supabase Auth or Auth0)

### Phase 3: Push Notifications (Week 5-6)

1. Integrate Firebase Cloud Messaging
2. User notification preferences
3. Background sync for PWA

### Phase 4: Mobile Apps (Future)

1. React Native or Flutter app
2. App store submission
3. Native push notifications

---

## Part 6: Security Considerations

### API Security

```python
# Secure the ingestion endpoint
@app.post("/api/v1/alerts")
async def receive_alert(
    alert: AlertSchema,
    api_key: str = Header(..., alias="X-API-Key")
):
    if not verify_api_key(api_key):
        raise HTTPException(401, "Invalid API key")

    # Process alert
    await store_alert(alert)
    await broadcast_to_subscribers(alert)
```

### Data Protection

#### Transport Security (HTTPS)

All communication must use TLS 1.2 or higher:

| Connection | Requirement |
| ---------- | ----------- |
| Home server → Cloud API | HTTPS with TLS 1.3 |
| Cloud API → Database | TLS encrypted connection |
| Web app → API | HTTPS only, HSTS enabled |
| WebSocket connections | WSS (WebSocket Secure) |

**Cloudflare Configuration:**

```toml
# wrangler.toml - Force HTTPS
[vars]
ENVIRONMENT = "production"

# In Cloudflare dashboard:
# SSL/TLS → Always Use HTTPS: ON
# SSL/TLS → Minimum TLS Version: 1.2
# SSL/TLS → HTTP Strict Transport Security (HSTS): Enable
```

#### API Key Management

**Home Server to Cloud Communication:**

```python
import os
import hashlib
import hmac
import time

class SecureAPIClient:
    def __init__(self):
        self.api_key = os.environ["CLOUD_API_KEY"]
        self.api_secret = os.environ["CLOUD_API_SECRET"]

    def generate_signature(self, payload: str, timestamp: int) -> str:
        """Generate HMAC signature for request authentication."""
        message = f"{timestamp}.{payload}"
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature

    async def send_alert(self, alert: dict):
        timestamp = int(time.time())
        payload = json.dumps(alert)
        signature = self.generate_signature(payload, timestamp)

        headers = {
            "X-API-Key": self.api_key,
            "X-Timestamp": str(timestamp),
            "X-Signature": signature,
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.yourdomain.com/v1/alerts",
                content=payload,
                headers=headers
            )
            return response.status_code == 200
```

**Cloud-Side Signature Verification:**

```python
from fastapi import HTTPException, Header
import time

MAX_TIMESTAMP_DIFF = 300  # 5 minutes

@app.post("/api/v1/alerts")
async def receive_alert(
    request: Request,
    x_api_key: str = Header(...),
    x_timestamp: str = Header(...),
    x_signature: str = Header(...)
):
    # Verify API key exists
    api_secret = await get_secret_for_key(x_api_key)
    if not api_secret:
        raise HTTPException(401, "Invalid API key")

    # Prevent replay attacks
    timestamp = int(x_timestamp)
    if abs(time.time() - timestamp) > MAX_TIMESTAMP_DIFF:
        raise HTTPException(401, "Request expired")

    # Verify signature
    body = await request.body()
    expected_sig = hmac.new(
        api_secret.encode(),
        f"{timestamp}.{body.decode()}".encode(),
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(x_signature, expected_sig):
        raise HTTPException(401, "Invalid signature")

    # Process alert
    alert = json.loads(body)
    await store_and_broadcast(alert)
```

#### User Authentication (JWT)

**Token Structure:**

```python
from datetime import datetime, timedelta
import jwt

SECRET_KEY = os.environ["JWT_SECRET"]
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

def create_tokens(user_id: str, tier: str) -> dict:
    """Create access and refresh tokens."""
    access_payload = {
        "sub": user_id,
        "tier": tier,
        "type": "access",
        "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        "iat": datetime.utcnow()
    }

    refresh_payload = {
        "sub": user_id,
        "type": "refresh",
        "exp": datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
        "iat": datetime.utcnow()
    }

    return {
        "access_token": jwt.encode(access_payload, SECRET_KEY, algorithm=ALGORITHM),
        "refresh_token": jwt.encode(refresh_payload, SECRET_KEY, algorithm=ALGORITHM),
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

def verify_token(token: str) -> dict:
    """Verify and decode JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")
```

**Protected Endpoint:**

```python
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    token = credentials.credentials
    payload = verify_token(token)

    if payload.get("type") != "access":
        raise HTTPException(401, "Invalid token type")

    return {
        "user_id": payload["sub"],
        "tier": payload.get("tier", "free")
    }

@app.get("/api/v1/alerts/stream")
async def stream_alerts(user: dict = Depends(get_current_user)):
    # User is authenticated, check tier for access level
    if user["tier"] == "free":
        # Return delayed alerts
        pass
    else:
        # Return real-time alerts
        pass
```

#### Rate Limiting

Protect against abuse and ensure fair usage:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

# Rate limits by endpoint
RATE_LIMITS = {
    "auth": "5/minute",      # Login attempts
    "api": "100/minute",     # General API calls
    "alerts": "1000/minute", # Alert streaming (authenticated)
    "webhook": "10/minute"   # Incoming webhooks
}

@app.post("/api/v1/auth/login")
@limiter.limit("5/minute")
async def login(request: Request, credentials: LoginSchema):
    # Authentication logic
    pass

@app.get("/api/v1/alerts")
@limiter.limit("100/minute")
async def get_alerts(request: Request, user: dict = Depends(get_current_user)):
    # Return alerts based on user tier
    pass
```

**Tier-Based Rate Limits:**

| Tier | Alerts/min | API calls/min | WebSocket connections |
| ---- | ---------- | ------------- | --------------------- |
| Free | 10 | 30 | 1 |
| Basic | 100 | 100 | 2 |
| Pro | Unlimited | 500 | 5 |

#### Data Encryption at Rest

**Database Encryption:**

```sql
-- PostgreSQL: Enable encryption for sensitive columns
-- Using pgcrypto extension
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Encrypt sensitive user data
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) NOT NULL UNIQUE,
    -- Store hashed password, never plaintext
    password_hash VARCHAR(255) NOT NULL,
    -- Encrypt payment info if stored (better to use Stripe's vault)
    stripe_customer_id VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- For highly sensitive data, use column-level encryption
-- (though Stripe handles payment data, so you rarely need this)
```

**Environment Variables / Secrets:**

| Secret | Storage Method |
| ------ | -------------- |
| Database credentials | Cloudflare Workers Secrets / AWS Secrets Manager |
| JWT signing key | Environment variable (rotated quarterly) |
| Stripe API keys | Environment variable |
| Home server API key | Environment variable |

```bash
# Cloudflare Workers - set secrets via CLI
wrangler secret put JWT_SECRET
wrangler secret put STRIPE_SECRET_KEY
wrangler secret put DATABASE_URL
wrangler secret put HOME_SERVER_API_KEY
```

#### Client-Side Security

**Never expose in frontend code:**

- API secret keys
- Database connection strings
- JWT signing secrets
- Stripe secret key (only use publishable key)

**Secure token storage:**

```javascript
// Store tokens securely in browser
class TokenManager {
  // Use httpOnly cookies for refresh tokens (set by server)
  // Store access token in memory only (not localStorage)

  private accessToken: string | null = null;

  setAccessToken(token: string) {
    this.accessToken = token;
    // Set expiry timer to refresh before expiration
    const payload = JSON.parse(atob(token.split('.')[1]));
    const expiresIn = payload.exp * 1000 - Date.now() - 60000; // Refresh 1 min early
    setTimeout(() => this.refreshToken(), expiresIn);
  }

  async refreshToken() {
    // Refresh token is in httpOnly cookie, sent automatically
    const response = await fetch('/api/v1/auth/refresh', {
      method: 'POST',
      credentials: 'include' // Include cookies
    });
    const data = await response.json();
    this.setAccessToken(data.access_token);
  }

  getAccessToken(): string | null {
    return this.accessToken;
  }
}
```

#### Input Validation and Sanitization

```python
from pydantic import BaseModel, validator, EmailStr
import re

class UserSignup(BaseModel):
    email: EmailStr
    password: str

    @validator('password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain a digit')
        return v

class AlertQuery(BaseModel):
    symbol: str
    limit: int = 50

    @validator('symbol')
    def valid_symbol(cls, v):
        # Only allow alphanumeric stock symbols
        if not re.match(r'^[A-Z]{1,5}$', v.upper()):
            raise ValueError('Invalid stock symbol')
        return v.upper()

    @validator('limit')
    def valid_limit(cls, v):
        if v < 1 or v > 500:
            raise ValueError('Limit must be between 1 and 500')
        return v
```

#### Audit Logging

Track security-relevant events:

```python
import logging
from datetime import datetime

# Structured logging for security events
security_logger = logging.getLogger("security")

class SecurityEvent:
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    TOKEN_REFRESH = "token_refresh"
    PERMISSION_DENIED = "permission_denied"
    RATE_LIMITED = "rate_limited"
    API_KEY_INVALID = "api_key_invalid"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"

async def log_security_event(
    event_type: str,
    user_id: str = None,
    ip_address: str = None,
    details: dict = None
):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event_type,
        "user_id": user_id,
        "ip": ip_address,
        "details": details or {}
    }
    security_logger.info(json.dumps(log_entry))

    # Store in database for analysis
    await db.security_logs.insert_one(log_entry)

# Usage
@app.post("/api/v1/auth/login")
async def login(request: Request, credentials: LoginSchema):
    ip = request.client.host

    user = await authenticate(credentials.email, credentials.password)
    if not user:
        await log_security_event(
            SecurityEvent.LOGIN_FAILED,
            ip_address=ip,
            details={"email": credentials.email}
        )
        raise HTTPException(401, "Invalid credentials")

    await log_security_event(
        SecurityEvent.LOGIN_SUCCESS,
        user_id=user.id,
        ip_address=ip
    )
    return create_tokens(user.id, user.tier)
```

#### Compliance Considerations

| Regulation | Applies If | Key Requirements |
| ---------- | ---------- | ---------------- |
| **GDPR** | EU customers | Data deletion rights, consent, breach notification |
| **CCPA** | California customers | Disclosure of data collection, opt-out rights |
| **PCI DSS** | Storing payment data | Use Stripe/Paddle instead (they handle compliance) |
| **SOC 2** | Enterprise customers | May be requested; use compliant cloud providers |

**Privacy Policy Must Include:**

- What data you collect (email, usage patterns, alerts viewed)
- How data is used (service delivery, analytics)
- Third parties (Stripe, analytics providers)
- Data retention period
- User rights (deletion, export)
- Contact information

**Data Deletion Endpoint:**

```python
@app.delete("/api/v1/user/me")
async def delete_account(user: dict = Depends(get_current_user)):
    user_id = user["user_id"]

    # Cancel Stripe subscription
    await cancel_stripe_subscription(user_id)

    # Delete user data
    await db.users.delete_one({"_id": user_id})
    await db.alert_preferences.delete_many({"user_id": user_id})
    await db.security_logs.update_many(
        {"user_id": user_id},
        {"$set": {"user_id": "DELETED"}}  # Anonymize, don't delete audit logs
    )

    await log_security_event(
        "account_deleted",
        user_id=user_id,
        details={"reason": "user_request"}
    )

    return {"message": "Account deleted"}
```

---

## Part 7: Quick Start Recommendation

**Minimum Viable Product (MVP):**

1. **Cloudflare Workers + D1 + R2**
   - Free tier covers initial usage
   - No server management

2. **Simple React PWA on Cloudflare Pages**
   - Real-time via SSE
   - Mobile-friendly responsive design

3. **Modify home server to POST alerts**
   ```python
   import httpx

   async def send_to_cloud(alert: dict):
       async with httpx.AsyncClient() as client:
           await client.post(
               "https://your-api.workers.dev/alerts",
               json=alert,
               headers={"X-API-Key": os.environ["CLOUD_API_KEY"]}
           )
   ```

This gets you a working web interface with minimal cost and complexity, which you can iterate on based on user feedback.

---

## Summary

| Decision | Recommendation | Reason |
|----------|----------------|--------|
| **Display Platform** | PWA first | Single codebase, no app store, push support |
| **Cloud Provider** | Cloudflare | Generous free tier, no egress fees |
| **Database** | PostgreSQL (Supabase or D1) | JSON support, real-time subscriptions |
| **Real-time** | SSE + Push Notifications | Simple, reliable, mobile-friendly |
| **Architecture** | Home server → Cloud API → Clients | Keep alert generation local, distribute via cloud |
