//! Multi-line SQL editor widget with syntax highlighting.
//!
//! Custom ratatui widget that provides a text editing experience with:
//! - Multi-line editing (Enter inserts newline)
//! - Cursor movement (arrows, Home/End)
//! - SQL syntax highlighting (keywords blue, identifiers green, literals yellow, numbers magenta)
//! - Integration with autocomplete popup (Tab triggers completion)

use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};

/// SQL keyword list for syntax highlighting (uppercase canonical forms).
const SQL_KEYWORDS: &[&str] = &[
    "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "IS", "NULL", "AS", "ON", "JOIN", "LEFT",
    "RIGHT", "INNER", "OUTER", "CROSS", "GROUP", "BY", "ORDER", "ASC", "DESC", "HAVING", "LIMIT",
    "OFFSET", "INSERT", "INTO", "VALUES", "UPDATE", "SET", "DELETE", "CREATE", "TABLE", "DROP",
    "ALTER", "INDEX", "DISTINCT", "COUNT", "SUM", "AVG", "MIN", "MAX", "BETWEEN", "LIKE", "EXISTS",
    "UNION", "ALL", "CASE", "WHEN", "THEN", "ELSE", "END", "CAST", "TRUE", "FALSE", "WITH",
    "DESCRIBE",
];

/// SQL aggregate/function names for highlighting.
const SQL_FUNCTIONS: &[&str] = &[
    "COUNT", "SUM", "AVG", "MIN", "MAX", "COALESCE", "NULLIF", "UPPER", "LOWER", "LENGTH", "TRIM",
    "SUBSTR", "CAST", "ABS", "ROUND", "CEIL", "FLOOR",
];

/// Token types for SQL syntax highlighting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    /// SQL keyword (SELECT, FROM, WHERE, etc.)
    Keyword,
    /// SQL function name (COUNT, SUM, etc.)
    Function,
    /// Identifier (column name, table name)
    Identifier,
    /// String literal ('...')
    StringLiteral,
    /// Numeric literal (123, 3.14)
    Number,
    /// Operator or punctuation (=, <, >, +, -, etc.)
    Operator,
    /// Whitespace
    Whitespace,
    /// Comment (-- or /* */)
    Comment,
}

/// A token with its text slice and kind.
#[derive(Debug, Clone)]
pub struct Token {
    pub text: String,
    pub kind: TokenKind,
}

/// Tokenize a SQL string for syntax highlighting.
pub fn tokenize_sql(input: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let ch = chars[i];

        // Whitespace
        if ch.is_whitespace() {
            let start = i;
            while i < len && chars[i].is_whitespace() {
                i += 1;
            }
            tokens.push(Token {
                text: chars[start..i].iter().collect(),
                kind: TokenKind::Whitespace,
            });
            continue;
        }

        // Single-line comment: --
        if ch == '-' && i + 1 < len && chars[i + 1] == '-' {
            let start = i;
            while i < len && chars[i] != '\n' {
                i += 1;
            }
            tokens.push(Token {
                text: chars[start..i].iter().collect(),
                kind: TokenKind::Comment,
            });
            continue;
        }

        // String literal: '...'
        if ch == '\'' {
            let start = i;
            i += 1; // skip opening quote
            while i < len && chars[i] != '\'' {
                if chars[i] == '\\' && i + 1 < len {
                    i += 1; // skip escaped char
                }
                i += 1;
            }
            if i < len {
                i += 1; // skip closing quote
            }
            tokens.push(Token {
                text: chars[start..i].iter().collect(),
                kind: TokenKind::StringLiteral,
            });
            continue;
        }

        // Number literal
        if ch.is_ascii_digit() || (ch == '.' && i + 1 < len && chars[i + 1].is_ascii_digit()) {
            let start = i;
            while i < len && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            tokens.push(Token {
                text: chars[start..i].iter().collect(),
                kind: TokenKind::Number,
            });
            continue;
        }

        // Identifier or keyword: starts with letter or underscore
        if ch.is_alphabetic() || ch == '_' {
            let start = i;
            while i < len && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            let upper = word.to_uppercase();

            let kind = if SQL_KEYWORDS.contains(&upper.as_str()) {
                TokenKind::Keyword
            } else if SQL_FUNCTIONS.contains(&upper.as_str()) {
                TokenKind::Function
            } else {
                TokenKind::Identifier
            };

            tokens.push(Token { text: word, kind });
            continue;
        }

        // Operators and punctuation (*, (, ), ,, ;, =, <, >, !, etc.)
        let start = i;
        // Multi-char operators: >=, <=, <>, !=, ||
        if i + 1 < len {
            let two: String = chars[i..i + 2].iter().collect();
            if matches!(two.as_str(), ">=" | "<=" | "<>" | "!=" | "||") {
                i += 2;
                tokens.push(Token {
                    text: two,
                    kind: TokenKind::Operator,
                });
                continue;
            }
        }
        i += 1;
        tokens.push(Token {
            text: chars[start..i].iter().collect(),
            kind: TokenKind::Operator,
        });
    }

    tokens
}

/// Get the color for a token kind.
pub fn token_color(kind: TokenKind) -> Color {
    match kind {
        TokenKind::Keyword => Color::Rgb(100, 140, 255), // blue
        TokenKind::Function => Color::Rgb(100, 140, 255), // blue (same as keywords)
        TokenKind::Identifier => Color::Rgb(80, 220, 120), // green
        TokenKind::StringLiteral => Color::Rgb(240, 220, 80), // yellow
        TokenKind::Number => Color::Rgb(220, 120, 255),  // magenta
        TokenKind::Operator => Color::Rgb(200, 200, 210), // light gray
        TokenKind::Whitespace => Color::Rgb(200, 200, 210), // light gray
        TokenKind::Comment => Color::Rgb(100, 100, 120), // dim gray
    }
}

/// Multi-line SQL editor state.
#[derive(Debug, Clone)]
pub struct EditorState {
    /// Lines of text in the editor buffer.
    pub lines: Vec<String>,
    /// Cursor row (0-indexed).
    pub cursor_row: usize,
    /// Cursor column (0-indexed, in chars not bytes).
    pub cursor_col: usize,
}

impl Default for EditorState {
    fn default() -> Self {
        Self::new()
    }
}

impl EditorState {
    /// Create an empty editor.
    pub fn new() -> Self {
        Self {
            lines: vec![String::new()],
            cursor_row: 0,
            cursor_col: 0,
        }
    }

    /// Get the full text content as a single string.
    pub fn text(&self) -> String {
        self.lines.join("\n")
    }

    /// Set the editor text from a string, resetting cursor to end.
    pub fn set_text(&mut self, text: &str) {
        self.lines = if text.is_empty() {
            vec![String::new()]
        } else {
            text.lines().map(|l| l.to_string()).collect()
        };
        if self.lines.is_empty() {
            self.lines.push(String::new());
        }
        self.cursor_row = self.lines.len() - 1;
        self.cursor_col = self.lines[self.cursor_row].chars().count();
    }

    /// Check if editor is empty.
    pub fn is_empty(&self) -> bool {
        self.lines.len() == 1 && self.lines[0].is_empty()
    }

    /// Insert a character at the cursor position.
    pub fn insert_char(&mut self, ch: char) {
        let line = &mut self.lines[self.cursor_row];
        let byte_pos = char_to_byte_pos(line, self.cursor_col);
        line.insert(byte_pos, ch);
        self.cursor_col += 1;
    }

    /// Insert a newline at the cursor position.
    pub fn insert_newline(&mut self) {
        let line = &self.lines[self.cursor_row];
        let byte_pos = char_to_byte_pos(line, self.cursor_col);
        let remainder = line[byte_pos..].to_string();
        self.lines[self.cursor_row] = line[..byte_pos].to_string();
        self.cursor_row += 1;
        self.lines.insert(self.cursor_row, remainder);
        self.cursor_col = 0;
    }

    /// Delete the character before the cursor (backspace).
    pub fn backspace(&mut self) {
        if self.cursor_col > 0 {
            let line = &mut self.lines[self.cursor_row];
            let byte_pos = char_to_byte_pos(line, self.cursor_col - 1);
            let end_pos = char_to_byte_pos(line, self.cursor_col);
            line.replace_range(byte_pos..end_pos, "");
            self.cursor_col -= 1;
        } else if self.cursor_row > 0 {
            // Join with previous line
            let current = self.lines.remove(self.cursor_row);
            self.cursor_row -= 1;
            self.cursor_col = self.lines[self.cursor_row].chars().count();
            self.lines[self.cursor_row].push_str(&current);
        }
    }

    /// Delete the character at the cursor (Delete key).
    pub fn delete_char(&mut self) {
        let line_len = self.lines[self.cursor_row].chars().count();
        if self.cursor_col < line_len {
            let line = &mut self.lines[self.cursor_row];
            let byte_pos = char_to_byte_pos(line, self.cursor_col);
            let end_pos = char_to_byte_pos(line, self.cursor_col + 1);
            line.replace_range(byte_pos..end_pos, "");
        } else if self.cursor_row + 1 < self.lines.len() {
            // Join with next line
            let next = self.lines.remove(self.cursor_row + 1);
            self.lines[self.cursor_row].push_str(&next);
        }
    }

    /// Move cursor left.
    pub fn move_left(&mut self) {
        if self.cursor_col > 0 {
            self.cursor_col -= 1;
        } else if self.cursor_row > 0 {
            self.cursor_row -= 1;
            self.cursor_col = self.lines[self.cursor_row].chars().count();
        }
    }

    /// Move cursor right.
    pub fn move_right(&mut self) {
        let line_len = self.lines[self.cursor_row].chars().count();
        if self.cursor_col < line_len {
            self.cursor_col += 1;
        } else if self.cursor_row + 1 < self.lines.len() {
            self.cursor_row += 1;
            self.cursor_col = 0;
        }
    }

    /// Move cursor up.
    pub fn move_up(&mut self) {
        if self.cursor_row > 0 {
            self.cursor_row -= 1;
            let line_len = self.lines[self.cursor_row].chars().count();
            self.cursor_col = self.cursor_col.min(line_len);
        }
    }

    /// Move cursor down.
    pub fn move_down(&mut self) {
        if self.cursor_row + 1 < self.lines.len() {
            self.cursor_row += 1;
            let line_len = self.lines[self.cursor_row].chars().count();
            self.cursor_col = self.cursor_col.min(line_len);
        }
    }

    /// Move cursor to start of line.
    pub fn move_home(&mut self) {
        self.cursor_col = 0;
    }

    /// Move cursor to end of line.
    pub fn move_end(&mut self) {
        self.cursor_col = self.lines[self.cursor_row].chars().count();
    }

    /// Get the word fragment at/before the cursor (for autocomplete).
    /// Returns (fragment, start_col) where start_col is where the fragment begins.
    pub fn word_at_cursor(&self) -> (String, usize) {
        let line = &self.lines[self.cursor_row];
        let chars: Vec<char> = line.chars().collect();
        let col = self.cursor_col.min(chars.len());

        // Walk backwards from cursor to find start of word
        let mut start = col;
        while start > 0 && (chars[start - 1].is_alphanumeric() || chars[start - 1] == '_') {
            start -= 1;
        }

        let fragment: String = chars[start..col].iter().collect();
        (fragment, start)
    }

    /// Replace the word at cursor with a completion string.
    pub fn apply_completion(&mut self, completion: &str, start_col: usize) {
        let line = &mut self.lines[self.cursor_row];
        let start_byte = char_to_byte_pos(line, start_col);
        let end_byte = char_to_byte_pos(line, self.cursor_col);
        line.replace_range(start_byte..end_byte, completion);
        self.cursor_col = start_col + completion.chars().count();
    }

    /// Render the editor content as syntax-highlighted Lines for ratatui.
    pub fn render_lines(&self) -> Vec<Line<'static>> {
        self.lines
            .iter()
            .map(|line| {
                if line.is_empty() {
                    return Line::from(Span::raw(" "));
                }
                let tokens = tokenize_sql(line);
                let spans: Vec<Span<'static>> = tokens
                    .into_iter()
                    .map(|tok| {
                        let color = token_color(tok.kind);
                        Span::styled(tok.text, Style::default().fg(color))
                    })
                    .collect();
                Line::from(spans)
            })
            .collect()
    }

    /// Render with cursor visible (adds cursor styling at cursor position).
    pub fn render_lines_with_cursor(&self) -> Vec<Line<'static>> {
        let mut result = Vec::with_capacity(self.lines.len());

        for (row_idx, line) in self.lines.iter().enumerate() {
            if row_idx == self.cursor_row {
                result.push(render_line_with_cursor(line, self.cursor_col));
            } else if line.is_empty() {
                result.push(Line::from(Span::raw(" ")));
            } else {
                let tokens = tokenize_sql(line);
                let spans: Vec<Span<'static>> = tokens
                    .into_iter()
                    .map(|tok| {
                        let color = token_color(tok.kind);
                        Span::styled(tok.text, Style::default().fg(color))
                    })
                    .collect();
                result.push(Line::from(spans));
            }
        }

        result
    }
}

/// Render a single line with a cursor at the given column.
fn render_line_with_cursor(line: &str, cursor_col: usize) -> Line<'static> {
    let chars: Vec<char> = line.chars().collect();
    let n = chars.len();

    if n == 0 {
        // Empty line: show cursor as a block character
        return Line::from(Span::styled(
            "\u{2588}",
            Style::default()
                .fg(Color::Rgb(255, 255, 255))
                .add_modifier(Modifier::BOLD),
        ));
    }

    // Tokenize the entire line for syntax highlighting
    let tokens = tokenize_sql(line);

    // Build spans char-by-char so we can override the cursor character's style
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut char_idx = 0;

    for tok in &tokens {
        let tok_chars: Vec<char> = tok.text.chars().collect();
        let base_color = token_color(tok.kind);

        for &ch in &tok_chars {
            if char_idx == cursor_col {
                // Cursor character: reverse video
                spans.push(Span::styled(
                    ch.to_string(),
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Rgb(255, 255, 255))
                        .add_modifier(Modifier::BOLD),
                ));
            } else {
                spans.push(Span::styled(
                    ch.to_string(),
                    Style::default().fg(base_color),
                ));
            }
            char_idx += 1;
        }
    }

    // If cursor is past end of line, append a cursor block
    if cursor_col >= n {
        spans.push(Span::styled(
            "\u{2588}",
            Style::default()
                .fg(Color::Rgb(255, 255, 255))
                .add_modifier(Modifier::BOLD),
        ));
    }

    Line::from(spans)
}

/// Convert a char-based column index to a byte position in a string.
fn char_to_byte_pos(s: &str, char_col: usize) -> usize {
    s.char_indices()
        .nth(char_col)
        .map(|(byte_pos, _)| byte_pos)
        .unwrap_or(s.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_select() {
        let tokens = tokenize_sql("SELECT id FROM users");
        assert_eq!(tokens.len(), 7); // SELECT, ws, id, ws, FROM, ws, users
        assert_eq!(tokens[0].kind, TokenKind::Keyword);
        assert_eq!(tokens[0].text, "SELECT");
        assert_eq!(tokens[2].kind, TokenKind::Identifier);
        assert_eq!(tokens[2].text, "id");
        assert_eq!(tokens[4].kind, TokenKind::Keyword);
        assert_eq!(tokens[4].text, "FROM");
        assert_eq!(tokens[6].kind, TokenKind::Identifier);
        assert_eq!(tokens[6].text, "users");
    }

    #[test]
    fn test_tokenize_where_with_number() {
        let tokens = tokenize_sql("WHERE amount > 100");
        assert_eq!(tokens[0].kind, TokenKind::Keyword);
        assert_eq!(tokens[2].kind, TokenKind::Identifier);
        assert_eq!(tokens[2].text, "amount");
        assert_eq!(tokens[4].kind, TokenKind::Operator);
        assert_eq!(tokens[4].text, ">");
        assert_eq!(tokens[6].kind, TokenKind::Number);
        assert_eq!(tokens[6].text, "100");
    }

    #[test]
    fn test_tokenize_string_literal() {
        let tokens = tokenize_sql("WHERE region = 'Europe'");
        let lit = tokens
            .iter()
            .find(|t| t.kind == TokenKind::StringLiteral)
            .unwrap();
        assert_eq!(lit.text, "'Europe'");
    }

    #[test]
    fn test_tokenize_function() {
        let tokens = tokenize_sql("SELECT count(*) FROM t");
        // count is in both SQL_KEYWORDS and SQL_FUNCTIONS -- keyword takes priority
        assert!(tokens[0].kind == TokenKind::Keyword); // SELECT
                                                       // count is checked as keyword first
        let count_tok = &tokens[2];
        assert_eq!(count_tok.text, "count");
        assert!(count_tok.kind == TokenKind::Keyword || count_tok.kind == TokenKind::Function);
    }

    #[test]
    fn test_tokenize_comment() {
        let tokens = tokenize_sql("SELECT 1 -- comment here");
        let comment = tokens
            .iter()
            .find(|t| t.kind == TokenKind::Comment)
            .unwrap();
        assert_eq!(comment.text, "-- comment here");
    }

    #[test]
    fn test_tokenize_multichar_operator() {
        let tokens = tokenize_sql("a >= 10");
        let op = tokens
            .iter()
            .find(|t| t.kind == TokenKind::Operator && t.text.len() > 1)
            .unwrap();
        assert_eq!(op.text, ">=");
    }

    #[test]
    fn test_editor_new() {
        let e = EditorState::new();
        assert!(e.is_empty());
        assert_eq!(e.cursor_row, 0);
        assert_eq!(e.cursor_col, 0);
        assert_eq!(e.text(), "");
    }

    #[test]
    fn test_editor_insert_and_text() {
        let mut e = EditorState::new();
        e.insert_char('S');
        e.insert_char('E');
        e.insert_char('L');
        assert_eq!(e.text(), "SEL");
        assert_eq!(e.cursor_col, 3);
    }

    #[test]
    fn test_editor_newline() {
        let mut e = EditorState::new();
        e.set_text("SELECT *");
        e.cursor_col = 8;
        e.insert_newline();
        assert_eq!(e.lines.len(), 2);
        assert_eq!(e.lines[0], "SELECT *");
        assert_eq!(e.lines[1], "");
        assert_eq!(e.cursor_row, 1);
        assert_eq!(e.cursor_col, 0);
    }

    #[test]
    fn test_editor_backspace() {
        let mut e = EditorState::new();
        e.set_text("abc");
        e.cursor_col = 3;
        e.backspace();
        assert_eq!(e.text(), "ab");
        assert_eq!(e.cursor_col, 2);
    }

    #[test]
    fn test_editor_backspace_join_lines() {
        let mut e = EditorState::new();
        e.set_text("line1\nline2");
        e.cursor_row = 1;
        e.cursor_col = 0;
        e.backspace();
        assert_eq!(e.lines.len(), 1);
        assert_eq!(e.text(), "line1line2");
    }

    #[test]
    fn test_editor_movement() {
        let mut e = EditorState::new();
        e.set_text("abc\ndef");
        e.cursor_row = 0;
        e.cursor_col = 1;

        e.move_right();
        assert_eq!(e.cursor_col, 2);

        e.move_left();
        assert_eq!(e.cursor_col, 1);

        e.move_down();
        assert_eq!(e.cursor_row, 1);
        assert_eq!(e.cursor_col, 1);

        e.move_up();
        assert_eq!(e.cursor_row, 0);
        assert_eq!(e.cursor_col, 1);

        e.move_home();
        assert_eq!(e.cursor_col, 0);

        e.move_end();
        assert_eq!(e.cursor_col, 3);
    }

    #[test]
    fn test_editor_word_at_cursor() {
        let mut e = EditorState::new();
        e.set_text("SELECT re");
        e.cursor_row = 0;
        e.cursor_col = 9; // after "re"

        let (fragment, start) = e.word_at_cursor();
        assert_eq!(fragment, "re");
        assert_eq!(start, 7);
    }

    #[test]
    fn test_editor_apply_completion() {
        let mut e = EditorState::new();
        e.set_text("SELECT re");
        e.cursor_row = 0;
        e.cursor_col = 9;

        let (_, start) = e.word_at_cursor();
        e.apply_completion("region", start);
        assert_eq!(e.text(), "SELECT region");
        assert_eq!(e.cursor_col, 13);
    }

    #[test]
    fn test_editor_render_lines() {
        let mut e = EditorState::new();
        e.set_text("SELECT id FROM t");
        let lines = e.render_lines();
        assert_eq!(lines.len(), 1);
        // Should have multiple spans (syntax colored)
        assert!(lines[0].spans.len() > 1);
    }

    #[test]
    fn test_editor_delete_char() {
        let mut e = EditorState::new();
        e.set_text("abc");
        e.cursor_row = 0;
        e.cursor_col = 1;
        e.delete_char();
        assert_eq!(e.text(), "ac");
    }

    #[test]
    fn test_char_to_byte_pos() {
        assert_eq!(char_to_byte_pos("hello", 0), 0);
        assert_eq!(char_to_byte_pos("hello", 3), 3);
        assert_eq!(char_to_byte_pos("hello", 5), 5);
    }

    #[test]
    fn test_token_colors_not_default() {
        // Ensure all token kinds produce actual RGB colors
        for kind in [
            TokenKind::Keyword,
            TokenKind::Function,
            TokenKind::Identifier,
            TokenKind::StringLiteral,
            TokenKind::Number,
            TokenKind::Operator,
            TokenKind::Whitespace,
            TokenKind::Comment,
        ] {
            match token_color(kind) {
                Color::Rgb(_, _, _) => {} // good
                _ => panic!("Expected Rgb for {:?}", kind),
            }
        }
    }
}
