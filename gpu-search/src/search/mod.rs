// Search engine module

pub mod binary;
pub mod channel;
pub mod content;
pub mod ignore;
pub mod orchestrator;
pub mod ranking;
pub mod streaming;
pub mod types;

pub use binary::BinaryDetector;
pub use channel::{search_channel, SearchChannel, SearchReceiver};
pub use self::ignore::GitignoreFilter;
pub use orchestrator::SearchOrchestrator;
pub use ranking::{rank_file_matches, rank_content_matches, truncate_results};
pub use types::*;
