# Database Schema and Search Response Reference

## Database Row Example

Here's a complete example of a single row from the `influencer_profiles` table in the LanceDB database:

### Basic Profile Information
- **id**: `45713773225` (Profile ID)
- **account**: `barodian_boy_` (Account Handle)
- **profile_name**: `𝐌𝐫.𝐯𝐞𝐞𝐫𝐩𝐚𝐥𝐬𝐢𝐧𝐡 𝐫𝐚𝐭𝐡𝐨𝐝` (Display Name)
- **full_name**: `𝐌𝐫.𝐯𝐞𝐞𝐫𝐩𝐚𝐥𝐬𝐢𝐧𝐡 𝐫𝐚𝐭𝐡𝐨𝐝` (Full Name)
- **profile_url**: `https://instagram.com/barodian_boy_`
- **biography**: `𝕄𝕚𝤸𝒪 𝕦𝕙 𝕡𝕒𝕡𝕒\n|| कृष्णं सदा सहायते ||🦚\n🦚जहाँ कृष्ण हैं, वहाँ राधा का नाम, और मेरे दिल में वही दोनों का धाम ❤️\n👻 @veer_v44`

### Account Status & Verification
- **is_private**: `False`
- **is_verified**: `False`
- **is_business_account**: `False`
- **is_professional_account**: `True`
- **is_joined_recently**: `False`
- **has_channel**: `False`

### Metrics & Engagement
- **followers**: `6705`
- **following**: `1255`
- **posts_count**: `622`
- **avg_engagement**: `2.667`
- **highlights_count**: `67`

### Business Information
- **business_category_name**: `Photographer`
- **category_name**: `Photographer`
- **business_address**: (empty)
- **external_url**: (empty)

### LLM Analysis Results
- **llm_processed**: `True`
- **individual_vs_org_score**: `2` (0=Individual, 10=Organization)
- **generational_appeal_score**: `5` (0=Low Gen Z appeal, 10=High Gen Z appeal)
- **professionalization_score**: `0` (0=Casual, 10=Highly professional)
- **relationship_status_score**: `0` (0=Single, 10=Relationship-focused)
- **keyword1**: `Bhakti`
- **keyword2**: `Krishna`
- **keyword3**: `Radha`
- **keyword4**: `Kali`
- **keyword5**: `Ganesh`
- **keyword6**: `Pavagadh`
- **keyword7**: `Vadodara`
- **keyword8**: `Champaner`
- **keyword9**: `Devotional`
- **keyword10**: `Spiritual`

### Content Analysis (JSON Fields)
- **posts**: JSON array containing 12 post objects with captions, likes, dates, and hashtags
- **highlights**: (empty)
- **related_accounts**: JSON array containing 74 related account objects
- **post_hashtags**: JSON array of hashtags used in posts

### Metadata
- **partner_id**: `45713773225.0`
- **email_address**: (empty)
- **timestamp**: (empty)
- **lance_db_id**: Auto-generated unique identifier for database operations
- **source_batch**: Identifies which LLM processing batch this record came from

## Vector Database Columns

The database also includes vector embeddings for search:
- **keyword_vector**: Vector embedding of extracted keywords (768 dimensions)
- **profile_vector**: Vector embedding of profile information (768 dimensions) 
- **content_vector**: Vector embedding of post content (768 dimensions)

## Search Response Format

When performing a vector search, the response returns a complete JSON array with all database fields plus search-specific scoring. Here's an example response for the query "fashion influencers":

```json
[
  {
    "lance_db_id": 47369,
    "account": "dailybutie",
    "fbid": 17841440589550288,
    "id": 40562304985,
    "followers": 104981,
    "posts_count": 744,
    "is_business_account": true,
    "is_professional_account": true,
    "is_verified": false,
    "avg_engagement": 1.3588,
    "external_url": "[\"https://bit.ly/3LJgJ8l\"]",
    "biography": "Follow me ❤️\nDaily posts 😍\nShop now 👇",
    "business_category_name": null,
    "category_name": "Ομορφιά, αισθητική και προσωπική φροντίδα",
    "post_hashtags": null,
    "following": 24.0,
    "posts": "[{\"caption\":\"CONTACT US\",\"comments\":11,\"content_type\":\"GraphImage\",\"datetime\":\"2023-10-18T22:17:18.000Z\",\"id\":\"3216590064854553708\",\"image_url\":\"https://instagram.fpfo1-1.fna.fbcdn.net/...\",\"likes\":160,\"url\":\"https://www.instagram.com/p/Cyjn58rtNRs\",\"video_url\":null}]",
    "profile_image_link": "https://instagram.fpfo1-1.fna.fbcdn.net/v/t51.2885-19/117791303_193302589076957_7608030807185085502_n.jpg...",
    "profile_url": "https://instagram.com/dailybutie",
    "profile_name": "Follow me for daily updates ❤️",
    "highlights_count": 0.0,
    "highlights": null,
    "full_name": "Follow me for daily updates ❤️",
    "is_private": false,
    "bio_hashtags": null,
    "business_email": null,
    "phone_number": null,
    "url": "https://instagram.com/dailybutie",
    "is_joined_recently": false,
    "has_channel": false,
    "partner_id": 40562304985.0,
    "business_address": null,
    "related_accounts": null,
    "email_address": null,
    "individual_vs_org_score": 4,
    "generational_appeal_score": 7,
    "professionalization_score": 0,
    "relationship_status_score": 0,
    "keyword1": "dailyupdates",
    "keyword2": "shop",
    "keyword3": "streetwear",
    "keyword4": "tshirts",
    "keyword5": "hoodies",
    "keyword6": "link_in_bio",
    "keyword7": "fashion",
    "keyword8": "brand",
    "keyword9": "influencer",
    "keyword10": "marketing",
    "llm_processed": true,
    "source_batch": "batch_results_003.jsonl",
    "keyword_text": "dailyupdates shop streetwear tshirts hoodies link_in_bio fashion brand influencer marketing",
    "profile_text": "Follow me ❤️\nDaily posts 😍\nShop now 👇 macro influencer gen z trendy modern slang emojis viral casual personal authentic unsponsored clearly single individual",
    "content_text_sample": "CONTACT US Follow me @dailybutie for daily updates    \n\nVia  jensin.camille on   Tk Follow me @dailybutie for daily updates...",
    "_distance": 0.48084816336631775,
    "keyword_score": 0.6752886789735173,
    "profile_score": 0.0,
    "content_score": 0.0,
    "combined_score": 0.2228452640612607,
    "query": "fashion influencers"
  }
]
```

### Search-Specific Fields
- **_distance**: Raw vector distance from query (lower = more similar)
- **keyword_score**: Similarity score for keyword vector (0-1, higher = more similar)
- **profile_score**: Similarity score for profile vector (0-1, higher = more similar)
- **content_score**: Similarity score for content vector (0-1, higher = more similar)
- **combined_score**: Weighted combination of all three scores (0-1, higher = more similar)
- **query**: The original search query used
- **keyword_text**: Concatenated keywords for search
- **profile_text**: Enhanced profile text with inferred characteristics
- **content_text_sample**: Sample of post content for search

## Database Statistics
- **Total records**: ~100,000 influencer profiles
- **LLM processed**: Variable percentage depending on batch processing completion
- **Vector dimensions**: 768 (using sentence-transformers/all-mpnet-base-v2)
- **Average followers**: Varies widely from micro to macro influencers
- **Verified accounts**: Small percentage of total database
- **Business accounts**: Significant portion of database