//! # Analysis
//! A framework for analyzing the memory and computation costs of training
//! CNN models on MCUs and generating a sparse update strategy to meet the
//! requirements.

/// Config for running the analysis tool
pub mod config;
/// Parsing of json files into CNN Models
/// and represent these CNN models in a format that can
/// be used for analysis
pub mod model_representation;
/// Generating a sparse update scheme for training a CNN
/// (represented using the framework)
/// based on the requirements
pub mod scheme_generation;
/// Output handling (displaying and TODO - conversion to other formats)
/// of generated schemes
pub mod scheme_representation;
