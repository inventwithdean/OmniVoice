use unicode_general_category::GeneralCategory::*;

pub enum ScriptCategory {
    Cjk,
    Hangul,
    Kana,
    Ethiopic,
    Yi,
    Indic,
    ThaiLao,
    KhmerMyanmar,
    Arabic,
    Hebrew,
    Latin,
    Cryillic,
    Greek,
    Armenian,
    Georgian,
    Punctuation,
    Space,
    Digit,
    Mark,
    Default,
}

impl ScriptCategory {
    pub fn weight(&self) -> f32 {
        match self {
            ScriptCategory::Cjk => 3.0,
            ScriptCategory::Hangul => 2.5,
            ScriptCategory::Kana => 2.2,
            ScriptCategory::Ethiopic => 3.0,
            ScriptCategory::Yi => 3.0,
            ScriptCategory::Indic => 1.8,
            ScriptCategory::ThaiLao => 1.5,
            ScriptCategory::KhmerMyanmar => 1.8,
            ScriptCategory::Arabic => 1.5,
            ScriptCategory::Hebrew => 1.5,
            ScriptCategory::Latin => 1.0,
            ScriptCategory::Cryillic => 1.0,
            ScriptCategory::Greek => 1.0,
            ScriptCategory::Armenian => 1.0,
            ScriptCategory::Georgian => 1.0,
            ScriptCategory::Punctuation => 0.5,
            ScriptCategory::Space => 0.2,
            ScriptCategory::Digit => 3.5,
            ScriptCategory::Mark => 0.0,
            ScriptCategory::Default => 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RuleDurationEstimator {}

impl RuleDurationEstimator {
    fn get_char_weight(char: char) -> ScriptCategory {
        match char {
            'A'..='Z' => ScriptCategory::Latin,
            ' ' => ScriptCategory::Space,
            c => match unicode_general_category::get_general_category(c) {
                NonspacingMark | SpacingMark | EnclosingMark => ScriptCategory::Mark,
                SpaceSeparator | LineSeparator | ParagraphSeparator => ScriptCategory::Space,
                DecimalNumber | LetterNumber | OtherNumber => ScriptCategory::Digit,
                DashPunctuation | OpenPunctuation | ClosePunctuation | FinalPunctuation
                | InitialPunctuation | ConnectorPunctuation | MathSymbol | CurrencySymbol
                | ModifierSymbol | OtherSymbol | OtherPunctuation => ScriptCategory::Punctuation,
                _ => Self::get_letter_script_category(c),
            },
        }
    }

    fn get_letter_script_category(char: char) -> ScriptCategory {
        match char as u32 {
            0x0000..=0x02AF => ScriptCategory::Latin,
            0x02B0..=0x03FF => ScriptCategory::Greek,
            0x0400..=0x052F => ScriptCategory::Cryillic,
            0x0530..=0x058F => ScriptCategory::Armenian,
            0x0590..=0x05FF => ScriptCategory::Hebrew,
            0x0600..=0x08FF => ScriptCategory::Arabic,
            0x0900..=0x0DFF => ScriptCategory::Indic,
            0x0E00..=0x0EFF => ScriptCategory::ThaiLao,
            0x0F00..=0x0FFF => ScriptCategory::Indic,
            0x1000..=0x109F => ScriptCategory::KhmerMyanmar,
            0x10A0..=0x10FF => ScriptCategory::Georgian,
            0x1100..=0x11FF => ScriptCategory::Hangul,
            0x1200..=0x139F => ScriptCategory::Ethiopic,
            0x13A0..=0x177F => ScriptCategory::Default,
            0x1780..=0x17FF => ScriptCategory::KhmerMyanmar,
            0x1800..=0x18FF => ScriptCategory::Default,
            0x1900..=0x19DF => ScriptCategory::Indic,
            0x19E0..=0x19FF => ScriptCategory::KhmerMyanmar,
            0x1A00..=0x1AAF => ScriptCategory::Indic,
            0x1AB0..=0x1BFF => ScriptCategory::Indic,
            0x1C00..=0x1C7F => ScriptCategory::Indic,
            0x1C80..=0x1C8F => ScriptCategory::Cryillic,
            0x1C90..=0x1CBF => ScriptCategory::Georgian,
            0x1CC0..=0x1CFF => ScriptCategory::Indic,
            0x1D00..=0x1DBF => ScriptCategory::Latin,
            0x1DC0..=0x1DFF => ScriptCategory::Default,
            0x1E00..=0x1EFF => ScriptCategory::Latin,
            0x1F00..=0x30FF => ScriptCategory::Kana,
            0x3100..=0x312F => ScriptCategory::Cjk,
            0x3130..=0x318F => ScriptCategory::Hangul,
            0x3190..=0x9FFF => ScriptCategory::Cjk,
            0xA000..=0xA4CF => ScriptCategory::Yi,
            0xA4D0..=0xA63F => ScriptCategory::Default,
            0xA640..=0xA69F => ScriptCategory::Cryillic,
            0xA6A0..=0xA6FF => ScriptCategory::Default,
            0xA700..=0xA7FF => ScriptCategory::Latin,
            0xA800..=0xA82F => ScriptCategory::Indic,
            0xA830..=0xA87F => ScriptCategory::Default,
            0xA880..=0xA95F => ScriptCategory::Indic,
            0xA960..=0xA97F => ScriptCategory::Hangul,
            0xA980..=0xA9DF => ScriptCategory::Indic,
            0xA9E0..=0xA9FF => ScriptCategory::KhmerMyanmar,
            0xAA00..=0xAA5F => ScriptCategory::Indic,
            0xAA60..=0xAA7F => ScriptCategory::KhmerMyanmar,
            0xAA80..=0xAAFF => ScriptCategory::Indic,
            0xAB00..=0xAB2F => ScriptCategory::Ethiopic,
            0xAB30..=0xAB6F => ScriptCategory::Latin,
            0xAB70..=0xABBF => ScriptCategory::Default,
            0xABC0..=0xABFF => ScriptCategory::Indic,
            0xAC00..=0xD7AF => ScriptCategory::Hangul,
            0xD7B0..=0xFAFF => ScriptCategory::Cjk,
            0xFB00..=0xFDFF => ScriptCategory::Arabic,
            0xFE00..=0xFE6F => ScriptCategory::Default,
            0xFE70..=0xFEFF => ScriptCategory::Arabic,
            0xFF00..=0xFFEF => ScriptCategory::Latin,
            0x20000..=u32::MAX => ScriptCategory::Cjk,
            _ => ScriptCategory::Default,
        }
    }

    pub fn calculate_total_weight(text: &str) -> f32 {
        text.chars()
            .map(|c| Self::get_char_weight(c).weight())
            .sum()
    }

    pub fn estimate_duration(
        &self,
        target_text: &str,
        ref_text: &str,
        ref_duration: f32,
        low_threshold: Option<f32>,
        boost_strength: f32,
    ) -> f32 {
        if ref_duration <= 0.0 || ref_text.is_empty() {
            return 0.0;
        }

        let ref_weight = Self::calculate_total_weight(ref_text);
        if ref_weight == 0.0 {
            return 0.0;
        }
        let speed_factor = ref_weight / ref_duration;
        let target_weight = Self::calculate_total_weight(target_text);

        let estimated_duration = target_weight / speed_factor;

        if let Some(threshold) = low_threshold {
            if estimated_duration < threshold {
                let alpha = 1.0 / boost_strength;
                return threshold * (estimated_duration / threshold).powf(alpha);
            }
        }
        estimated_duration
    }
}
