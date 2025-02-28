from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import torch
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen
import asyncio
from transformers import MarianMTModel, MarianTokenizer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import langdetect
import io
from rapidfuzz import process


model_name = 'Helsinki-NLP/opus-mt-ko-en'
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)


# CLIP 모델과 프로세서 로딩
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")


genre_map = {
    "Ambient": [
        "calm", "peaceful", "relaxing", "tranquil", "meditative", "serene", 
        "atmospheric", "spacious", "dreamy", "minimalistic", "natural", "slow",
        "experimental", "spacey", "mellow", "contemplative", "ambient soundscapes", 
        "soothing", "reflective", "introspective", "gentle", "immersive", "flowing",
        "quiet", "delicate", "soft", "still", "airy", "harmonic", "hypnotic", "vast",
        "serene landscapes", "cosmic", "celestial", "textural", "chill", "emotive",
        "deep", "surreal", "therapeutic", "placid", "tranquilizing", "abstract",
        "harmonious", "enveloping", "meditative drones", "evolving", "layered", 
        "nature-inspired", "shimmering", "spectral", "otherworldly", "peaceful vibes", 
        "echoing", "resonant", "tonal", "cinematic", "sparse", "distant", "moody", "light", 
        "fluid", "subdued", "abstract tones", "quietude", "hazy", "reverberant", 
        "subdued beauty", "mystical", "transcendent", "atmospheric pads", "minimal tones", 
        "unstructured", "evolving patterns", "liquid", "ephemeral", "ghostly",
        "introspective moods", "zen", "introspective soundscapes", "slow-moving", "glacial", 
        "meditative atmospheres", "field recordings", "ambient drones", "subtle", "calm rhythm", 
        "textural depth", "placid tones", "celestial movements", "sonic meditation", "twilight", 
        "crystalline", "nocturnal", "surreal landscapes"
    ],
    "Classical": [
        "orchestral", "symphonic", "baroque", "romantic", "timeless", "grand", "majestic",
        "refined", "elegant", "virtuosic", "dramatic", "harmonious", "sophisticated", 
        "neo-classical", "elevated", "formal", "intricate", "melodic", "structured",
        "traditional", "chamber music", "concerto", "sonata", "fugue", "operatic", "polyphonic", 
        "expressive", "lyrical", "classical guitar", "piano sonata", "symphonic poem", 
        "pastoral", "choral", "sacred", "liturgical", "cantata", "stately", "rondo",
        "waltz", "serenade", "minuet", "overture", "impressionistic", "modern classical", 
        "tonal", "atonal", "avant-garde", "historical", "detailed", "pure", "thematic",
        "classical variations", "concerto grosso", "partita", "suite", "elegy", "requiem",
        "aria", "florid", "counterpoint", "instrumental", "canon", "ricercar", 
        "orchestral suite", "meditative", "ethereal", "regal", "poised", "delicate", 
        "sonorous", "sweeping", "narrative", "poignant", "transcendent", 
        "expressive dynamics", "improvisatory", "authentic", "scholarly", "sublime",
        "melancholic", "dramatic arcs", "historic resonance", "intricate motifs",
        "balanced", "phrased", "noble", "expansive", "emotional depth", 
        "classical aesthetics", "grandiosity", "clarity", "thematic development", 
        "disciplined", "legacy", "immortalized"
    ],
    "Jazz": [
        "improvisational", "swinging", "smooth", "bebop", "cool", "fusion", "urban",
        "expressive", "sophisticated", "lounge", "bluesy", "vibrant", "groovy", 
        "intellectual", "rhythmic", "syncopated", "soulful", "free jazz", "modal",
        "avant-garde", "hot jazz", "jazz standards", "big band", "ragtime", "brassy", 
        "jazzy", "club", "harmonic", "scat", "lyrical", "melodic", "upbeat", "dynamic", 
        "nuanced", "swing era", "piano-driven", "horns", "bassline", "jazzy chords",
        "cool tones", "jazzy improvisation", "soul-jazz", "jazzy grooves", "vocal jazz", 
        "instrumental", "quartet", "trio", "jazz ballad", "introspective", "high-energy",
        "bebop tempo", "bebop runs", "relaxed", "chill jazz", "traditional jazz",
        "modern jazz", "jazz fusion", "funky", "blended styles", "nu-jazz", 
        "jazz club vibe", "upbeat swing", "cool solos", "Latin jazz", "jazz melodies", 
        "chord progressions", "polyrhythmic", "jazzy themes", "melancholic jazz",
        "percussive", "walking bass", "jazzy atmosphere", "saxophone solos", "vocal scat",
        "laid-back", "groove-heavy", "syncopation", "extended harmony", "jazz phrasing", 
        "smooth dynamics", "jazzy rhythm", "emotive playing", "unique voice",
        "jazzy dynamics", "nuanced solos", "bold improvisation", "creative phrasing",
        "jazz storytelling", "jazz heritage", "historic roots", "jazz history"
    ],
    "Blues": [
        "melancholic", "soulful", "gritty", "emotional", "yearning", "raw", "downbeat",
        "moody", "traditional", "folk-inspired", "bluesy", "authentic", "repetitive",
        "improvised", "storytelling", "heartfelt", "electric blues", "acoustic blues",
        "Delta blues", "Chicago blues", "Texas blues", "classic blues", "slide guitar",
        "12-bar blues", "minor key", "pentatonic scale", "call and response", 
        "blues riffs", "blues licks", "blue notes", "slow blues", "boogie-woogie", 
        "blues shuffle", "blues ballad", "harmonica", "gravelly vocals", "lament", 
        "blues guitar", "sorrowful", "gritty rhythm", "fingerpicking", "guitar bends", 
        "blues solo", "swing feel", "groove-driven", "emotive lyrics", "expressive", 
        "rural", "urban", "soul-driven", "pain", "struggle", "resilience", "crooning",
        "whiskey-soaked", "haunting", "introspective", "dark", "uplifting", "hopeful", 
        "rich", "classic riffs", "deep tone", "shuffling beat", "intimate", "personal", 
        "traditional feel", "jazzy blues", "funky blues", "modern blues", "roots blues", 
        "southern blues", "blues spirit", "emotive storytelling", "sorrowful melodies", 
        "bittersweet", "rough edges", "raw sound", "gravelly timbre", "wailing",
        "crying guitar", "desperate", "somber", "slow tempo", "soul-stirring",
        "authentic voice", "unique tone", "timeless"
    ],
    "Rock": [
        "intense", "guitar-driven", "energetic", "rebellious", "anthemic", "hard-hitting", 
        "alternative", "classic", "indie", "progressive", "heavy", "experimental", 
        "raw", "dynamic", "powerful", "grungy", "distorted guitars", "stadium rock",
        "punk-influenced", "hard rock", "psychedelic", "garage rock", "emo", 
        "new wave", "post-punk", "riff-driven", "anthem rock", "headbanging", 
        "melodic rock", "modern rock", "retro rock", "fusion rock", "post-rock",
        "shoegaze", "guitar solos", "power chords", "catchy hooks", "fast-paced", 
        "raw energy", "raspy vocals", "high octane", "arena sound", "uplifting",
        "angsty", "fiery", "passionate", "electric riffs", "punchy", "live band feel",
        "emotive", "heartfelt lyrics", "upbeat", "introspective", "hard-edged", 
        "dynamic arrangements", "layered sound", "impactful", "earthy tones", 
        "vintage vibes", "modern edge", "thunderous drums", "powerful choruses",
        "driving rhythm", "moody", "rebellion", "defiant spirit", "nostalgic", 
        "youthful energy", "anthemic melodies", "epic soundscapes", "underground rock",
        "alternative edge", "timeless classics", "adrenaline-fueled", "crunchy riffs", 
        "gritty undertones", "full-bodied sound", "expressive vocals", "storytelling lyrics", 
        "dynamic shifts", "compelling"
    ],
    "Metal": [
        "heavy", "dark", "aggressive", "powerful", "chaotic", "thrashing", "melodic",
        "doom", "black", "death", "industrial", "symphonic", "avant-garde", "epic", 
        "intense", "ritualistic", "unrelenting", "guttural vocals", "double bass", 
        "distorted guitars", "blast beats", "shredding solos", "minor scales",
        "dark atmospheres", "palm muting", "high-energy", "headbanging", "extreme tempos", 
        "technical", "progressive metal", "groove", "nu-metal", "power metal", 
        "folk metal", "viking metal", "metalcore", "post-metal", "ambient metal", 
        "doom-laden", "sludgy", "stoner", "raw energy", "underground", "gothic", 
        "mythical themes", "warrior spirit", "epic storytelling", "operatic", 
        "ritualistic", "ferocious", "explosive", "firepower", "ominous", "haunting",
        "majestic", "devastating", "titanic riffs", "cinematic", "unrelenting pace",
        "macabre", "dissonant", "pummeling", "gritty", "relentless", "pioneering", 
        "apocalyptic", "otherworldly", "transcendent", "mythic narratives", 
        "feral energy", "cathartic", "crushing intensity", "harrowing", "menacing",
        "punishing", "visceral", "iconic", "untamed", "uncompromising", "dynamic",
        "searing solos", "virtuosic", "monumental"
    ],
    "Hip Hop": [
        "rhythmic", "beat-driven", "lyrical", "urban", "sharp", "edgy", 
        "trap", "conscious", "gangsta", "old-school", "boom bap", "modern", 
        "experimental", "political", "sampled", "streetwise", "gritty", 
        "authentic", "flow", "freestyle", "swagger", "cypher", "underground", 
        "energetic", "storytelling", "punchy", "raw", "melodic", "emotive", 
        "hard-hitting", "street beats", "turntable", "DJ scratch", "battle rap", 
        "poetic", "minimal", "grime", "bounce", "anthemic", "club-ready", 
        "vocal-driven", "versatile", "wordplay", "vibrant", "west coast", 
        "east coast", "trap soul", "hyphy", "bass-heavy", "jazzy beats", 
        "funky", "chopped and screwed", "ambient hip hop", "lo-fi hip hop", 
        "social commentary", "catchy hooks", "diss track", "flow state", 
        "autotune", "rap god", "underdog stories", "hustler", "street anthem", 
        "gangsta life", "introspective", "motivational", "regional styles", 
        "mainstream", "independent", "empowering", "melancholic", "fast-paced", 
        "dynamic", "layered", "cultural", "multi-syllabic", "fast flow", 
        "slow flow", "street poetry", "piano beats", "trap beats", "pop-infused", 
        "neo-soul vibes", "hardcore", "political consciousness", "club bangers", 
        "upbeat", "motivational lyrics", "lyrical complexity", "minimal beats", 
        "high-energy rap", "southern rap", "bass drops", "uplifting", "versatile flows"
    ],
    "Lo-fi": [
        "chill", "nostalgic", "cozy", "mellow", "soft", "relaxing", 
        "bedroom", "vintage", "warm", "hazy", "introspective", "peaceful", 
        "smooth", "laid-back", "lo-fi beats", "minimal", "ambient", "soothing", 
        "old cassette", "vinyl crackle", "jazzy vibes", "study music", 
        "soft drums", "calm piano", "guitar riffs", "melancholic", "reflective", 
        "acoustic textures", "downtempo", "instrumental", "dreamy", "meditative", 
        "bedroom producer", "intimate", "serene", "hypnotic", "lo-fi jazz", 
        "fuzzy", "lo-fi chillhop", "deep vibes", "nighttime", "soft melodies", 
        "vibey", "gentle rhythms", "emotional", "slow-paced", "muted beats", 
        "underground", "mood music", "minimalist", "aesthetic", "retro", 
        "vintage synthesizer", "muted tones", "soft ambiance", "calming tunes", 
        "DIY sound", "lo-fi edits", "personal", "handmade", "textured sounds", 
        "non-intrusive", "creative loops", "soft transitions", "rainy day", 
        "cloudy skies", "lo-fi soul", "bedroom vibes", "background music", 
        "tape-inspired", "lo-fi nostalgia", "old-school warmth", "gentle keys", 
        "muffled beats", "lo-fi acoustics", "sleepy", "chillwave", "tranquil", 
        "late night", "cozy evenings", "peaceful loops", "slow rhythms", 
        "light grooves", "soft vinyl tones", "warm fuzz", "casual vibes", 
        "artistic layers", "lo-fi dreams", "background vibes", "soft chords", 
        "reflective soundscapes", "comforting", "familiar", "relaxed energy"

    ],
    "Pop": [
        "catchy", "fun", "upbeat", "danceable", "lighthearted", "radio-friendly", 
        "modern", "bubblegum", "electropop", "anthemic", "mainstream", "melodic", 
        "polished", "accessible", "hook-laden", "vibrant", "feel-good", 
        "chart-topping", "sing-along", "energetic", "pop rock", "romantic", 
        "playful", "youthful", "glossy", "catchy lyrics", "smooth vocals", 
        "pop anthems", "dynamic", "radio hits", "synth-heavy", "electro-pop", 
        "pop fusion", "emotive", "power ballads", "heartfelt", "dance floor", 
        "stylish", "upbeat melodies", "festival vibes", "dreamy", "uplifting", 
        "celebratory", "feel-good vibes", "pop classics", "timeless", 
        "crossover hits", "commercial", "singer-songwriter pop", "infectious", 
        "pop energy", "high-production", "groovy", "summer vibes", "party-ready", 
        "vocal-driven", "catchy hooks", "top 40", "feel-good energy", 
        "positive", "bright tones", "high-energy", "dance-pop", "sing-along vibes", 
        "vocal harmonies", "modern anthems", "dance-floor ready", "love songs", 
        "optimistic", "bubbly", "chart-friendly", "catchy choruses", "shiny", 
        "guitar-pop", "synth-pop", "pop ballads", "mood-lifting", "airy", 
        "pop perfection", "pop bangers", "teen spirit", "inspirational pop", 
        "high-octane", "celebratory pop", "feel-good anthems", "fun-loving", 
        "youth anthem", "radio-smash", "contemporary", "pop trends"

    ],
    "Electronic": [
        "futuristic", "synth-heavy", "dynamic", "nightlife", "vibrant", 
        "downtempo", "house", "techno", "trance", "breakbeat", "experimental", 
        "cybernetic", "dance", "ambient", "progressive", "spacey", "synthwave"
    ],
    "Reggae": [
        "island-inspired", "groovy", "chill", "uplifting", "sunny", "laid-back", 
        "roots", "dancehall", "dub", "positive vibes", "relaxing", "ritualistic", 
        "smooth", "soulful", "rhythmic", "conscious", "feel-good"
    ],
    "Latin": [
        "fiery", "romantic", "lively", "danceable", "rhythmic", "celebratory", 
        "flamenco", "salsa", "bossa nova", "reggaeton", "tropical", "passionate", 
        "exotic", "vibrant", "festive", "spicy", "seductive"
    ],
    "Country": [
        "storytelling", "acoustic", "down-to-earth", "warm", "nostalgic", 
        "bluegrass", "western", "americana", "heartfelt", "rustic", "twangy", 
        "honky-tonk", "folksy", "romantic", "traditional", "outdoorsy"
    ],
    "EDM": [
        "party", "pulsating", "electrifying",
        "club", "future bass", "dubstep", "progressive house", "big room", 
         "trap", "bouncy", "vibrant", "hypnotic", "uplifting"
    ],
    "R&B": [
        "smooth", "romantic", "soulful", "sensual", "melodic", 
        "groovy", "neo-soul", "modern", "intimate", "heartfelt", "luxurious", 
        "slow jams", "emotional", "sophisticated", "rhythmic", "vocal-driven"
    ],
    "Soul": [
        "uplifting", "deep", "powerful", "authentic", "spiritual", 
        "gospel-inspired", "emotional", "expressive", "timeless", "heartwarming", 
        "smooth", "soothing", "energetic", "passionate", "warm", "resonant"
    ],
    "Punk": [
        "raw", "fast", "rebellious", "aggressive", "anti-establishment", 
        "DIY", "garage", "anarchistic", "energetic", "gritty", "loud", "direct", 
        "uncompromising", "punk rock", "subversive", "agitated"
    ],
    "Folk": [
        "natural", "traditional", "rustic", "acoustic", "organic", 
        "peaceful", "storytelling", "intimate", "roots", "cultural", "simple", 
        "heartfelt", "nostalgic", "pastoral", "authentic", "reflective"
    ],
    "Synthwave": [
        "retro", "80s-inspired", "futuristic", "neon", "cyberpunk", 
        "vintage", "nostalgic", "dreamy", "electronic", "atmospheric", "synth-heavy", 
        "catchy", "dynamic", "pulsating", "vibrant", "moody"
    ],
    "World": [
        "ethnic", "global", "tribal", "folkloric", "cultural", 
        "traditional", "roots", "eclectic", "diverse", "authentic", "spiritual", 
        "mystical", "regional", "ceremonial", "celebratory", "danceable"
    ],
    "Indie": [
        "alternative", "quirky", "unique", "raw", "creative", 
        "original", "experimental", "underground", "personal", "expressive", 
        "authentic", "DIY", "lo-fi", "contemplative", "melancholic", "dreamy"
    ],
    "Gospel": [
        "spiritual", "holy", "choir", "uplifting", "faith", 
        "soulful", "harmonic", "reverent", "powerful", "divine", "inspirational", 
        "praise", "joyful", "heavenly", "sacred", "heartfelt"
    ]
    #,
    # "Dream Pop": [
    #     "ethereal", "whimsical", "surreal", "airy", "floaty",
    #     "mystical", "dreamy", "nostalgic", "lush", "ambient", "delicate",
    #     "melodic", "introspective", "spacey", "otherworldly", "hypnotic"
    # ]
}



# 테마 매핑
theme_map = {
    "Nature": [
        "peaceful", "outdoors", "forest", "mountains", "beach", 
        "serene", "natural", "exploration", "sunrise", "wildlife", 
        "tranquil", "desert", "riverside", "rainforest", "snowy landscape", 
        "flower fields", "savanna", "waterfalls", "caves", "starry sky", 
        "volcanic landscapes", "coral reefs", "underwater world", "arctic tundra",
        "seasonal blooms", "echoing canyons", "glacial valleys", "stormy seas",
        "misty mornings", "rocky shores", "distant horizons", "lush jungles", 
        "silent lakes", "frozen lakes", "pine forests", "secluded beaches", 
        "mountain peaks", "rolling hills", "sunlit meadows", "rushing rivers"
    ],
    "city": [
        "city life", "modern", "nightlife", "industrial", "fast-paced", 
        "gritty", "cosmopolitan", "streets", "neon", "metropolitan", 
        "skyscrapers", "subway", "rooftop", "alleyways", "markets", 
        "abandoned buildings", "train stations", "cafes", "urban parks", 
        "graffiti walls", "night markets", "suburban sprawl", "construction zones",
        "skyscraper views", "downtown", "rush hour", "street food", "highways", 
        "billboards", "night lights", "back alleys", "loft apartments", "modern architecture"
    ],
    "Fantasy": [
        "mystical", "magical", "ethereal", "otherworldly", "imaginative", 
        "mythical", "fairy tale", "epic", "heroic", "legendary", 
        "enchanted forests", "dragons", "sorcery", "kingdoms", "quests", 
        "floating islands", "ancient ruins", "dark castles", "crystal caves", 
        "celestial realms", "underworld", "time portals", "mystic rivers", 
        "legendary heroes", "celestial guardians", "secret realms", "wandering spirits", 
        "distant kingdoms", "ancient prophecies", "magic portals", "otherworldly creatures", 
        "immortal beings"
    ],
    "Romance": [
        "love", "passion", "intimacy", "sensuality", "tenderness", 
        "affection", "longing", "heartfelt", "emotional", "devotion", 
        "starry nights", "candlelight", "blossoms", "letters", "embraces", 
        "secret rendezvous", "stolen glances", "wedding moments", "first dates", 
        "romantic walks", "serendipitous meetings", "destined lovers", 
        "moonlit nights", "rose petals", "cherry blossoms", "whispered promises", 
        "gentle kisses", "intimate moments", "falling in love", "unspoken words"
    ],
    "Adventure": [
        "exploration", "thrill", "journey", "wanderlust", "discovery", 
        "freedom", "danger", "heroic", "new horizons", "uncharted", 
        "treasure hunts", "mountain climbing", "desert expeditions", "deep sea", 
        "space travel", "jungle trekking", "polar expeditions", "ancient artifacts", 
        "lost cities", "daring escapes", "wild frontiers", "hidden trails",
        "hidden caves", "mythical quests", "sailing adventures", "river rafting", 
        "snowboard expeditions", "mysterious islands", "dangerous waters", "search for treasure"
    ],
    "Celebration": [
        "party", "joy", "festive", "lively", "cheerful", 
        "vibrant", "dancing", "uplifting", "happiness", "fun", 
        "parades", "fireworks", "birthdays", "weddings", "anniversaries", 
        "music festivals", "carnivals", "street parties", "reunions", 
        "victories", "grand openings", "community gatherings", "balloons", 
        "confetti", "cheers", "cheerful crowds", "celebratory toasts", "bright colors"
    ],
    "Mystery": [
        "enigmatic", "dark", "curious", "cryptic", "suspenseful", 
        "foreboding", "tense", "haunting", "unknown", "investigative", 
        "hidden clues", "foggy nights", "whispers", "locked doors", "abandoned places", 
        "ancient secrets", "coded messages", "shadowy figures", "secret societies", 
        "hidden passageways", "forbidden libraries", "puzzling ruins", 
        "mysterious symbols", "vanishing acts", "unsolved mysteries", "hidden agendas", 
        "whispering winds", "secret letters"
    ],
    "Energy": [
        "dynamic", "high-energy", "vibrant", "intense", "powerful", 
        "pulsating", "energetic", "bold", "fast-paced", "action-packed", 
        "electricity", "motorsports", "stadium", "adrenaline", "workout", 
        "racing", "competition", "sports", "victories", "extreme stunts", 
        "breakdancing", "cheering crowds", "roller coasters", "extreme sports", 
        "high-speed chases", "energy bursts", "fast action", "pulse-racing"
    ],
    "Classical": [
        "ancient", "medieval", "renaissance", "revolutionary", "nostalgic", 
        "timeless", "cultural", "traditional", "classic", "retro", 
        "artifacts", "ruins", "castles", "period costumes", "legends", 
        "wartime", "golden age", "dynasties", "mythology", "pioneering", 
        "folk traditions", "victorian elegance", "historical events", "ancient cities", 
        "warrior codes", "old civilizations", "forgotten empires", "vintage"
    ],
    "Electronic": [
        "futuristic", "space", "technology", "alien worlds", "cyberpunk", 
        "robotics", "dystopian", "utopian", "galaxies", "time travel", 
        "spaceships", "artificial intelligence", "quantum mechanics", "terraforming", 
        "virtual reality", "biotech", "cyborgs", "parallel universes", 
        "extraterrestrial civilizations", "neon cities", "interstellar wars", 
        "alien invasions", "space exploration", "dystopian societies", "mind control", 
        "robot revolutions", "neon lights", "galactic rebellions", "space stations"
    ],
    "Horror": [
        "spooky", "creepy", "haunted", "dark", "sinister", 
        "chilling", "macabre", "terrifying", "supernatural", "psychological", 
        "ghosts", "shadows", "graveyards", "monsters", "isolated places", 
        "ancient curses", "haunted forests", "eerie silence", "dark rituals", 
        "abandoned asylums", "unseen presences", "paranormal activity", 
        "twisted minds", "night terrors", "darkness", "evil spirits"
    ],
    # "Nature": [
    #     "spring", "summer", "autumn", "winter", "holidays",
    #     "festivals", "harvest", "snowfall", "sunshine", "rainy days",
    #     "New Year", "Halloween", "Christmas", "Valentine's Day", "Easter",
    #     "cherry blossoms", "autumn leaves", "snowmen", "summer breeze",
    #     "thanksgiving", "seasonal feasts", "holiday lights", "festive seasons",
    #     "harvest moon", "autumn pumpkins", "spring blossoms", "winter wonderland"
    # ],
    "War": [
        "battlefields", "armies", "strategy", "heroic sacrifice", "valor", 
        "siege", "trenches", "naval warfare", "air combat", "medics", 
        "resistance", "victory celebrations", "historic battles", "generals", 
        "weapons development", "warfront stories", "tactical maneuvers", 
        "combat zones", "military tactics", "wartime strategy", "defensive lines"
    ]
    # ,
    # "Dream": [
    #     "surreal", "imaginative", "phantasmagoric", "abstract", "flowing",
    #     "luminous", "infinite", "kaleidoscopic", "unreal", "ethereal",
    #     "floating", "shifting", "dreamscapes", "illusory", "transcendent",
    #     "mind-bending", "otherworldly visions", "impossible landscapes",
    #     "shifting realities", "lucid dreams", "visionary", "bizarre"
    # ]
}



# 분위기 매핑
mood_map = {
    "Relaxed": [
        "calm", "peaceful", "soothing", "tranquil", "gentle", 
        "laid-back", "cozy", "content", "serene", "quiet", 
        "meditative", "easygoing", "harmonious", "idyllic", "unwind", 
        "zen-like", "breezy", "mellow", "flowing", "light", 
        "casual", "unruffled", "soft", "effortless", "relaxed vibes"
    ],
    "Energetic": [
        "vibrant", "dynamic", "intense", "high-energy", "bold", 
        "pulsating", "electrifying", "excited", "fast-paced", "motivated", 
        "thrilling", "fiery", "explosive", "uplifting", "adrenaline-fueled", 
        "pumping", "high-octane", "amped", "exuberant", "spirited", 
        "unstoppable", "racing", "eager", "energetic chaos", "intense drive", 
        "non-stop", "high voltage", "pulse-quickening", "vibrant momentum"
    ],
    "Melancholic": [
        "sad", "nostalgic", "reflective", "introspective", "yearning", 
        "moody", "lonely", "wistful", "haunting", "somber", 
        "regretful", "aching", "mournful", "blue", "pensive", 
        "melancholy", "brooding", "bittersweet", "downhearted", "sorrowful", 
        "gloomy", "desolate", "melodramatic", "fragile", "understated sadness", 
        "heartbroken", "heavy-hearted", "wistful longing", "tragic beauty"
    ],
    "Joyful": [
        "happy", "uplifting", "cheerful", "fun", "positive", 
        "lighthearted", "excited", "playful", "bright", "elated", 
        "ecstatic", "gleeful", "content", "carefree", "jubilant", 
        "gleaming", "radiant", "sparkling", "overjoyed", "delighted", 
        "sunny", "euphoric", "chipper", "giggly", "bubbly", "upbeat", 
        "blissful", "carefree fun", "exuberant happiness", "delirious joy"
    ],
    "Tense": [
        "nervous", "anxious", "suspenseful", "uneasy", "foreboding", 
        "intense", "gritty", "on-edge", "ominous", "dark", 
        "claustrophobic", "chilling", "dreadful", "alarming", "tense", 
        "jittery", "conflicted", "troubled", "unnerving", "apprehensive", 
        "turbulent", "restless", "frantic", "panicked", "volatile", 
        "electric anxiety", "crushing pressure", "stressful tension", "unsettling"
    ],
    "Romantic": [
        "tender", "passionate", "sensual", "intimate", "affectionate", 
        "dreamy", "loving", "sweet", "emotional", "heartfelt", 
        "ardent", "devoted", "charming", "alluring", "yearning", 
        "whimsical", "magnetic", "flirtatious", "serenading", "ethereal love", 
        "longing", "romantic glow", "feminine mystique", "soft affection", 
        "amorous", "sensual embrace", "charmed", "romantic serenade"
    ],
    "Dark": [
        "sinister", "eerie", "haunting", "creepy", "macabre", 
        "menacing", "mysterious", "ominous", "brooding", "shadowy", 
        "bleak", "grim", "foreboding", "desolate", "cold", 
        "nightmarish", "ghostly", "tragic", "otherworldly", "ominous forewarning", 
        "gruesome", "hauntingly beautiful", "intense darkness", "gripping suspense"
    ],
    "Warm": [
        "cozy", "reassuring", "comforting", "safe", "nurturing", 
        "embracing", "gentle", "friendly", "supportive", "pleasant", 
        "healing", "inviting", "sunlit", "homely", "cherished", 
        "soft glow", "gentle warmth", "heartwarming", "radiating", 
        "hospitable", "welcoming", "intimate", "sun-kissed", "affectionate"
    ],
    "Playful": [
        "mischievous", "fun-loving", "quirky", "spirited", "cheeky", 
        "carefree", "humorous", "joking", "spontaneous", "excited", 
        "adventurous", "frolicsome", "whimsical", "eccentric", "silly", 
        "lively", "energetic fun", "gleeful chaos", "bubbly mischief", 
        "playful antics", "light-hearted fun", "cheerful spontaneity"
    ],
    # "Mysterious": [
    #     "enigmatic", "cryptic", "suspenseful", "haunting", "arcane",
    #     "esoteric", "uncanny", "curious", "otherworldly", "shadowy",
    #     "obscure", "intense", "puzzling", "unknown", "haunting beauty",
    #     "mystic wonder", "hidden secrets", "ethereal mystery",
    #     "dark intrigue", "esoteric allure", "bewitching", "cryptic beauty"
    # ],
    "Hopeful": [
        "optimistic", "aspiring", "uplifting", "encouraging", "bright", 
        "positive", "motivational", "reassuring", "promising", "renewing", 
        "uplifting calm", "inspiring dreams", "future-oriented", "empowered", "glimmers of hope", 
        "hopeful dreams", "renewed faith", "empowered vision", "inspirational light"
    ],
    "Epic": [
        "heroic", "grand", "monumental", "majestic", "powerful", 
        "dramatic", "legendary", "adventurous", "triumphant", "unstoppable", 
        "towering", "commanding", "sweeping", "immense", "historic", 
        "cosmic", "infinite", "game-changing", "mythical", "unstoppable force"
    ],
    "Chilling": [
        "spine-tingling", "cold", "ominous", "eerie", "haunting", 
        "ghostly", "unnerving", "darkly silent", "icy", "creeping dread", 
        "terrifying calm", "frozen fear", "isolating", "distant wails", 
        "chilling calm", "haunted silence", "creepy stillness", "cold embrace"
    ],
    "Peaceful": [
        "quiet", "calm", "serene", "tranquil", "gentle", 
        "harmonious", "restful", "soothing", "relaxing", "zen-like", 
        "soft murmurs", "stillness", "gentle flow", "healing", 
        "blissful quiet", "calming presence", "peaceful calm", "soft serenity", 
        "restorative silence", "natural peace", "inner peace", "relaxed harmony"
    ]
}







app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/audio_files", StaticFiles(directory="audio_files"), name="audio_files")

duration: int = 15
topk: int = 250
topp: float = 0.0
temperature: float = 1.0
cfg_coef: float = 3.0

class MusicRequest(BaseModel):
    text: str


# 비동기 번역 함수
async def translate_text(text: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _translate_text, text)

def _translate_text(text: str) -> str:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77  # 모델의 최대 길이에 맞춤
    )
    translated = model.generate(**inputs)
    result = tokenizer.decode(translated[0], skip_special_tokens=True)
    return result



# 비동기 음악 생성 함수
async def generate_music_async(translated_text: str) -> list:
    model_musicgen = MusicGen.get_pretrained('large')
    model_musicgen.set_generation_params(duration=duration, top_k=topk, top_p=topp, temperature=temperature, cfg_coef=cfg_coef)
    
    outputs = model_musicgen.generate(translated_text, progress=False)
    outputs = outputs.detach().cpu().float()

    output_files = []
    for idx, output in enumerate(outputs):
        temp_file_path = f"audio_files/output_{idx}.wav"
        audio_write(temp_file_path, output, model_musicgen.sample_rate, strategy="loudness", loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)
        output_files.append(temp_file_path)
    
    return output_files




# 이미지에서 설명 추출 (CLIP 사용)
def generate_image_description(image_bytes: io.BytesIO):
    image = Image.open(image_bytes)
    inputs = clip_processor(images=image, return_tensors="pt", padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    return image_features

# 음악 장르 추천 (CLIP 모델을 이용해 설명과 장르 비교)
def get_best_match(features, mapping):
    similarities = []
    for key, keywords in mapping.items():
        # 키워드 리스트를 하나의 문장으로 결합
        description = " ".join(keywords)
        inputs = clip_processor(text=description, return_tensors="pt", padding=True, truncation=True, max_length=77)
        with torch.no_grad():
            text_features = clip_model.get_text_features(**inputs)
        similarity = torch.cosine_similarity(features, text_features)
        similarities.append((key, similarity.item()))

    return max(similarities, key=lambda x: x[1])[0]  # 가장 높은 유사도를 가진 항목 반환

def recommend_music_details_from_image(image_bytes: io.BytesIO):
    image_features = generate_image_description(image_bytes)

    # 장르, 테마, 분위기 각각의 매칭 결과 도출
    genre = get_best_match(image_features, genre_map)
    theme = get_best_match(image_features, theme_map)
    mood = get_best_match(image_features, mood_map)

    return {
        "genre": genre,
        "theme": theme,
        "mood": mood
    }







@app.post("/generate-music/")
async def generate_music(request: MusicRequest):
    try:
        if not os.path.exists('audio_files'):
            os.mkdir('audio_files')

        print("music generation request : ", request)

        # 비동기적으로 번역을 진행하고, 완료될 때까지 기다림
        detected_language = langdetect.detect(request.text)

        if detected_language == 'en':
            translatedText = request.text
        else:
            translatedText = await translate_text(request.text)
            print("번역이 완료되었습니다 : ", translatedText)
        translatedText = [translatedText]

        # 번역된 텍스트를 기반으로 음악 생성
        output_files = await generate_music_async(translatedText)

        return {"file_paths": output_files[0]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-music-from-image/")
async def generate_music_from_image(image: UploadFile = File(...)):
    try:
        # 이미지 처리
        image_bytes = io.BytesIO(await image.read())
        music_details = recommend_music_details_from_image(image_bytes)

        # 매핑 결과를 텍스트로 변환
        translated_text = f"{music_details['genre']}, {music_details['theme']}, {music_details['mood']}"
        print("Generated Text for Music:", translated_text)

        # 음악 생성
        if not os.path.exists('audio_files'):
            os.mkdir('audio_files')
        output_files = await generate_music_async([translated_text])

        return {"file_paths": output_files[0]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
