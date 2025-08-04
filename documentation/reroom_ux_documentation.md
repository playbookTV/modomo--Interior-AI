# ReRoom: Complete UX Documentation

## 📱 Core User Flows

### **Primary User Flow: Photo to Purchase**
```
App Launch → Onboarding → Camera → Style Selection → Processing → Results → Shopping → Purchase
    ↓           ↓         ↓        ↓               ↓           ↓        ↓         ↓
  30s         60s       90s      15s             15s         3m      5m       External
```

### **Secondary Flows:**
- **Saved Rooms Management:** Browse → View → Edit → Reshare
- **Price Alerts:** Setup → Monitor → Notify → Purchase
- **Premium Upgrade:** Feature Gate → Paywall → Subscribe → Enhanced Features
- **Social Sharing:** Results → Share → Social Media → Viral Growth

---

## 🎨 Screen-by-Screen Breakdown

### **1. App Launch & Splash**

```
┌─────────────────────┐
│                     │
│    [ReRoom Logo]    │
│                     │
│   Snap. Style.      │
│      Save.          │
│                     │
│  [Loading Bar]      │
│                     │
│                     │
└─────────────────────┘
```

**Purpose:** Brand introduction, app loading
**Duration:** 2-3 seconds
**Elements:**
- ReRoom logo (animated)
- Tagline with price emphasis
- Loading indicator
- Background: Subtle gradient

---

### **2. Onboarding Flow (First Time Users)**

#### **2a. Welcome Screen**
```
┌─────────────────────┐
│     Skip     Get Started │
│                     │
│  [Hero Video Preview]│
│   Before/After Room │
│                     │
│ "Save £200-800 per  │
│  room makeover"     │
│                     │
│ • Professional Design│
│ • Best Price Discovery│ 
│ • 15-Second Results │
│                     │
│  [Continue Button]  │
└─────────────────────┘
```

#### **2b. Style Preference Quiz**
```
┌─────────────────────┐
│  Question 1 of 3    │
│                     │
│ "Which room style   │
│  speaks to you?"    │
│                     │
│ [Modern]  [Scandi]  │
│  [Image]   [Image]  │
│                     │
│ [Boho]    [Industrial]│
│  [Image]   [Image]  │
│                     │
│      [Next] →       │
└─────────────────────┘
```

#### **2c. Budget Range**
```
┌─────────────────────┐
│  Question 2 of 3    │
│                     │
│ "What's your typical │
│  room budget?"      │
│                     │
│ ○ £200-500         │
│ ○ £500-1,000       │
│ ○ £1,000-2,000     │
│ ○ £2,000+          │
│                     │
│ "We'll prioritize   │
│  the best deals"    │
│                     │
│      [Next] →       │
└─────────────────────┘
```

#### **2d. Permission Requests**
```
┌─────────────────────┐
│  Question 3 of 3    │
│                     │
│ "Ready to transform │
│   your first room?" │
│                     │
│   [Camera Icon]     │
│                     │
│ We need camera      │
│ access to analyze   │
│ your room photos    │
│                     │
│ [Allow Camera] ✓    │
│ [Maybe Later]       │
│                     │
│ [Enable Notifications]│
│ "Get price alerts"  │
└─────────────────────┘
```

---

### **3. Camera Screen (Core Functionality)**

```
┌─────────────────────┐
│ ×  [Flash] [Flip]   │
│                     │
│                     │
│  ┌─────────────────┐│
│  │                 ││ 
│  │   [Viewfinder]  ││
│  │                 ││
│  │ [Quality Guide] ││
│  │ [Room Frame]    ││
│  │                 ││
│  └─────────────────┘│
│                     │
│ "Point camera at    │
│  furniture you want │
│   to redesign"      │
│                     │
│     ○ [Capture]     │
│   [Gallery] [Tips]  │
└─────────────────────┘
```

**Key Interactive Elements:**

#### **Quality Guidance Overlay:**
```
Real-time feedback based on image analysis:
┌─────────────────────┐
│  ⚠️ "Need more light" │
│  ✅ "Perfect angle"   │
│  📏 "Move closer"     │
│  🎯 "Focus on furniture"│
└─────────────────────┘
```

#### **Room Detection Frame:**
```
Dynamic overlay showing detected furniture:
┌─────────────────────┐
│     [Sofa] ✓       │
│  [Coffee Table] ✓   │
│     [Rug] ✓        │
│                     │
│ Confidence: 89%     │
│ [Tap to Capture]    │
└─────────────────────┘
```

---

### **4. Photo Review & Crop**

```
┌─────────────────────┐
│  ←   Crop Photo   ✓ │
│                     │
│ ┌─────────────────┐ │
│ │  [Captured Photo] │ │
│ │                 │ │
│ │ [Crop Handles]  │ │
│ │                 │ │
│ │ [Grid Overlay]  │ │
│ └─────────────────┘ │
│                     │
│ 🔄 Rotate  📐 Square │
│                     │
│ Quality Score: 91%  │
│ ✅ "Great lighting!" │
│                     │
│   [Retake] [Use This]│
└─────────────────────┘
```

---

### **5. Style Selection**

```
┌─────────────────────┐
│  ←  Choose Style    │
│                     │
│ ┌─────────────────┐ │
│ │ [Horizontal Scroll]│ │
│ │                 │ │
│ │ [Modern] [Scandi]│ │
│ │ [Boho] [Industrial]│ │
│ │ [Minimalist]    │ │
│ └─────────────────┘ │
│                     │
│ Selected: Modern    │
│                     │
│ "Clean lines,       │
│  neutral colors,    │
│  contemporary feel" │
│                     │
│ Est. Cost: £450-750 │
│ Avg. Savings: £287  │
│                     │
│  [Transform Room] → │
└─────────────────────┘
```

**Style Cards Design:**
```
Each style card shows:
┌─────────┐
│ [Image] │ Preview of style
│ Modern  │ Style name
│ £350-650│ Price range
│ Save £200│ Potential savings
└─────────┘
```

---

### **6. AI Processing Screen**

```
┌─────────────────────┐
│  ←  Processing...   │
│                     │
│ ┌─────────────────┐ │
│ │                 │ │
│ │ [Original Photo]│ │
│ │                 │ │
│ │ [Pulse Animation]│ │
│ │                 │ │
│ └─────────────────┘ │
│                     │
│ ████████████░░░ 78% │
│                     │
│ Analyzing room      │
│ layout...           │
│                     │
│ Next: Finding best  │
│ prices across stores│
│                     │
│ ⏱️ 8 seconds left    │
└─────────────────────┘
```

**Processing Stages:**
1. "Analyzing your room..." (0-25%)
2. "Understanding layout..." (25-50%)
3. "Applying modern style..." (50-75%)
4. "Finding best prices..." (75-95%)
5. "Almost ready!" (95-100%)

---

### **7. Results Screen (Key Revenue Driver)**

```
┌─────────────────────┐
│ ← Share 💾 ⚙️        │
│                     │
│ ┌─────────────────┐ │
│ │                 │ │
│ │ [Before] [After]│ │
│ │  Photo Comparison │ │
│ │                 │ │
│ │ [Slide to compare]│ │
│ └─────────────────┘ │
│                     │
│ 🎉 You'll save £347! │
│                     │
│ Total Cost: £543    │
│ vs £890 elsewhere   │
│                     │
│ [Shop This Look] 🛒 │
│ [Save Room] ⭐      │
│ [Try Another Style] │
└─────────────────────┘
```

**Tap-to-Explore Mode:**
```
When user taps on furniture in result:
┌─────────────────────┐
│     [Result Image]  │
│                     │
│ 👆 Tapped: Sofa     │
│                     │
│ ┌─────────────────┐ │
│ │ Modern Sofa     │ │
│ │ [Product Image] │ │
│ │ £299 at Amazon  │ │
│ │ Was: £399       │ │
│ │ You save: £100  │ │
│ │                 │ │
│ │ [Buy Now] [Compare]│ │
│ │ [Save for Later]│ │
│ └─────────────────┘ │
└─────────────────────┘
```

---

### **8. Shopping Experience**

#### **8a. Product List View**
```
┌─────────────────────┐
│ ← Modern Living Room│
│                     │
│ Total: £543 (Save £347)│
│ ┌─────────────────┐ │
│ │ ☑️ Sofa          │ │
│ │ [Image] £299    │ │
│ │ Amazon • Save £100│ │
│ │ [Buy] [Remove]  │ │
│ └─────────────────┘ │
│ ┌─────────────────┐ │
│ │ ☑️ Coffee Table  │ │
│ │ [Image] £89     │ │
│ │ Temu • Save £45 │ │
│ │ [Buy] [Remove]  │ │
│ └─────────────────┘ │
│ ┌─────────────────┐ │
│ │ ☑️ Area Rug      │ │
│ │ [Image] £155    │ │
│ │ Argos • Save £89│ │
│ │ [Buy] [Remove]  │ │
│ └─────────────────┘ │
│ [Buy Everything] £543│
└─────────────────────┘
```

#### **8b. Individual Product Detail**
```
┌─────────────────────┐
│ ←  Modern Sofa      │
│                     │
│ ┌─────────────────┐ │
│ │                 │ │
│ │ [Product Gallery]│ │
│ │ [Swipe Images]  │ │
│ │                 │ │
│ └─────────────────┘ │
│                     │
│ Grey 3-Seater Sofa │
│ ⭐⭐⭐⭐⭐ 4.6 (1,247)  │
│                     │
│ 🏷️ £299 (Was £399)   │
│ 💰 You save: £100    │
│                     │
│ 📦 Free delivery     │
│ 🚚 Arrives Tuesday   │
│                     │
│ [Similar Items] ↔️   │
│                     │
│ [Add to Cart] [Compare]│
│ [Save for Later]    │
└─────────────────────┘
```

#### **8c. Price Comparison**
```
┌─────────────────────┐
│ ← Compare Prices    │
│                     │
│ Grey 3-Seater Sofa │
│                     │
│ ⭐ Amazon UK £299    │
│   ✅ Best Price     │
│   Free delivery     │
│   [Choose This] →   │
│                     │
│ John Lewis £349     │
│   2-year warranty   │
│   [View Details]    │
│                     │
│ Wayfair £375        │
│   White glove setup │
│   [View Details]    │
│                     │
│ Very.co.uk £399     │
│   Buy now pay later │
│   [View Details]    │
│                     │
│ 💡 Price Alert: ON  │
│ We'll notify if cheaper│
└─────────────────────┘
```

---

### **9. Premium Upgrade Screens**

#### **9a. Feature Gate (3 Free Renders)**
```
┌─────────────────────┐
│     ⭐ Upgrade      │
│                     │
│ You've used 3/3     │
│ free room designs   │
│                     │
│ ┌─────────────────┐ │
│ │ Price Pro       │ │
│ │ £12.99/month    │ │
│ │                 │ │
│ │ ✅ Unlimited rooms│ │
│ │ ✅ Price alerts   │ │
│ │ ✅ Deal forecasts │ │
│ │ ✅ Priority speed │ │
│ │ ✅ Savings analytics│ │
│ └─────────────────┘ │
│                     │
│ "Pays for itself   │
│  with one purchase" │
│                     │
│ [Start Free Trial] │
│ [Maybe Later]      │
└─────────────────────┘
```

#### **9b. Price Alert Setup**
```
┌─────────────────────┐
│ ← Price Alert Setup │
│                     │
│ Get notified when   │
│ saved items drop    │
│ in price           │
│                     │
│ ┌─────────────────┐ │
│ │ 🔔 Alert when:   │ │
│ │                 │ │
│ │ ○ Any price drop │ │
│ │ ● 10% or more   │ │
│ │ ○ 20% or more   │ │
│ │ ○ Custom amount │ │
│ └─────────────────┘ │
│                     │
│ 📱 Notification:    │
│ ☑️ Push notifications│
│ ☑️ Email alerts     │
│ ☐ SMS alerts (+£1/mo)│
│                     │
│ [Save Preferences] │
└─────────────────────┘
```

---

### **10. Saved Rooms & Management**

#### **10a. Saved Rooms Grid**
```
┌─────────────────────┐
│ ← Saved Rooms (5)   │
│                     │
│ ┌─────┐ ┌─────┐     │
│ │Room1│ │Room2│     │
│ │ [📷] │ │ [📷] │     │
│ │Living│ │Bedroom    │
│ │Modern│ │Scandi│    │
│ │£543 │ │£789 │     │
│ └─────┘ └─────┘     │
│ ┌─────┐ ┌─────┐     │
│ │Room3│ │Room4│     │
│ │ [📷] │ │ [📷] │     │
│ │Kitchen││Dining│    │
│ │Industrial│Boho│   │
│ │£321 │ │£456 │     │
│ └─────┘ └─────┘     │
│ ┌─────┐             │
│ │Room5│ [+ New]     │
│ │ [📷] │  Room       │
│ │Office│             │
│ │Mini. │             │
│ │£234 │             │
│ └─────┘             │
└─────────────────────┘
```

#### **10b. Saved Room Detail**
```
┌─────────────────────┐
│ ← Living Room • Edit│
│                     │
│ ┌─────────────────┐ │
│ │ [Room Image]    │ │
│ │ Modern Style    │ │
│ │ Created: 2 days │ │
│ │ Total: £543     │ │
│ │ Potential: £347 │ │
│ └─────────────────┘ │
│                     │
│ Recent Price Changes│
│ 📉 Sofa: £299→£279  │
│ 📈 Table: £89→£95   │
│ ➡️ Rug: £155 (same) │
│                     │
│ New Total: £529     │
│ 💰 Extra savings: £14│
│                     │
│ [Shop Updated Prices]│
│ [Share Room] [Delete]│
│ [Try New Style]     │
└─────────────────────┘
```

---

### **11. Profile & Settings**

```
┌─────────────────────┐
│ ← Profile           │
│                     │
│ [Avatar] Emma       │
│ emma@email.com      │
│ Price Pro Member ⭐ │
│                     │
│ 💰 Total Saved      │
│    £1,247          │
│                     │
│ 🏠 Rooms Designed   │
│    7 rooms         │
│                     │
│ ⚙️ Settings         │
│ └ Notifications     │
│ └ Privacy          │
│ └ Account          │
│                     │
│ 📊 Savings Analytics│
│ 🎁 Refer Friends    │
│ 📞 Help & Support   │
│ 📝 About           │
│                     │
│ [Sign Out]         │
└─────────────────────┘
```

---

## 📱 Navigation Patterns

### **Primary Navigation (Bottom Tab Bar)**
```
┌─────────────────────┐
│                     │
│   [Main Content]    │
│                     │
│                     │
├─────────────────────┤
│ 🏠    📷    💾    👤 │
│Home Camera Saved Profile│
└─────────────────────┘
```

### **Secondary Navigation Patterns**
- **Header Back Arrow:** ← for hierarchical navigation
- **Close X:** For modal dismissal
- **Settings Gear:** ⚙️ for configuration screens
- **Share Icon:** For social sharing functionality

---

## 🎨 Visual Design System

### **Color Palette**
```
Primary Colors:
• Deep Black (#000000) - Headers, primary text
• Pure White (#FFFFFF) - Backgrounds, cards
• Electric Blue (#0066FF) - CTAs, interactive elements

Secondary Colors:
• Forest Green (#10B981) - Success, savings indicators
• Amber Gold (#F59E0B) - Warnings, alerts
• Soft Grey (#F3F4F6) - Inactive elements, borders
• Deep Red (#EF4444) - Errors, urgent actions
```

### **Typography**
```
Headings: SF Pro Display (iOS) / Roboto (Android)
• H1: 28px Bold
• H2: 24px Semibold  
• H3: 20px Medium

Body Text: SF Pro Text (iOS) / Roboto (Android)
• Large: 18px Regular
• Medium: 16px Regular
• Small: 14px Regular
• Caption: 12px Regular

Interactive: 16px Medium (minimum touch target)
```

### **Component Library**

#### **Buttons**
```
Primary CTA:
┌─────────────────┐
│  Shop This Look │ (Blue background, white text)
│     £543        │
└─────────────────┘

Secondary:
┌─────────────────┐
│   Save Room     │ (White background, blue text)
└─────────────────┘

Destructive:
┌─────────────────┐
│   Remove Item   │ (Red background, white text)
└─────────────────┘
```

#### **Cards**
```
Product Card:
┌─────────────────┐
│ [Product Image] │
│ Product Name    │
│ £299 (Was £399) │
│ Save £100 ✅     │
│                 │
│ [Buy Now]       │
└─────────────────┘

Room Card:
┌─────────────────┐
│ [Room Image]    │
│ Living Room     │
│ Modern Style    │
│ Total: £543     │
│ Saved: £347     │
└─────────────────┘
```

#### **Forms & Inputs**
```
Text Input:
┌─────────────────┐
│ Enter email...  │ (Grey border, focus = blue)
└─────────────────┘

Toggle Switch:
Budget Alerts ●—○ ON

Price Slider:
£100 ●————————○ £1000
```

---

## 🔄 Interaction Patterns

### **Gestures**
- **Tap:** Primary interaction, button presses, item selection
- **Long Press:** Context menus, item manipulation
- **Swipe Left/Right:** Photo galleries, style browsing
- **Pinch to Zoom:** Photo review, result examination
- **Pull to Refresh:** Reload price data, update rooms

### **Animations**
- **Screen Transitions:** Slide from right (forward), slide to right (back)
- **Loading States:** Pulse animation for processing, progress bars
- **Success States:** Checkmark animation, savings counter
- **Micro-interactions:** Button press feedback, toggle switches

### **Haptic Feedback**
- **Success:** Light impact (purchase completion, save)
- **Warning:** Medium impact (price increase alert)
- **Error:** Heavy impact (failed processing, network error)
- **Selection:** Light tap (style selection, toggle)

---

## 📊 UX Metrics & Testing

### **Key UX Metrics**
- **Time to First Render:** <60 seconds from app open
- **Processing Completion Rate:** >92% successful renders
- **Purchase Conversion:** >70% from result view to purchase
- **Return Usage:** >60% return within 7 days

### **A/B Testing Framework**
- **Onboarding Flow:** Test different value propositions
- **Style Selection:** Compare grid vs carousel layouts
- **Results Presentation:** Test savings emphasis vs design emphasis
- **Purchase Flow:** Optimize button placement and messaging

### **Accessibility Considerations**
- **Voice Over:** Full screen reader support
- **Dynamic Type:** Scalable font sizes
- **Color Contrast:** WCAG AA compliance
- **Touch Targets:** Minimum 44px tap areas
- **Reduced Motion:** Respect system preferences

---

## 🚀 Progressive Enhancement

### **Core Experience (Must Work)**
- Photo capture and upload
- Basic style transformation
- Product price display
- External purchase links

### **Enhanced Experience (Nice to Have)**
- Real-time price updates
- Advanced animations
- Offline saved rooms
- Social sharing integration

### **Premium Experience (Subscription Features)**
- Unlimited transforms
- Priority processing
- Advanced analytics
- Price prediction

This UX documentation provides a complete blueprint for building ReRoom's mobile experience, prioritizing the price-discovery value proposition while maintaining an intuitive, delightful user interface.