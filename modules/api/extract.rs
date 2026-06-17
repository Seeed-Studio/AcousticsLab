//! Extractor wrappers reshaping stock axum's plain-text rejection bodies into
//! the `{error, code}` envelope ([`ApiError::Bad`]) for a uniform wire surface.

use axum::extract::FromRequest;
use axum::extract::FromRequestParts;
use axum::extract::rejection::{JsonRejection, QueryRejection};
use axum::http::request::Parts;
use serde::de::DeserializeOwned;

use crate::api::error::ApiError;

/// Body cap for control-plane JSON routes (train/convert tiny config structs):
/// bounds the pre-`serde_json` buffer so bogus oversize bodies can't pin axum's
/// 2 MiB default resident per request. Uploads set their own larger limits.
pub(crate) const CONTROL_JSON_BODY_LIMIT: usize = 64 * 1024;

#[derive(Debug, Clone, Copy, Default)]
pub struct ApiJson<T>(pub T);

impl<T, S> FromRequest<S> for ApiJson<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request(
        req: axum::http::Request<axum::body::Body>,
        state: &S,
    ) -> Result<Self, Self::Rejection> {
        match axum::Json::<T>::from_request(req, state).await {
            Ok(axum::Json(value)) => Ok(ApiJson(value)),
            Err(rej) => Err(map_json_rejection(rej)),
        }
    }
}

fn map_json_rejection(rej: JsonRejection) -> ApiError {
    bad_extract("invalid JSON body", rej)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ApiQuery<T>(pub T);

impl<T, S> FromRequestParts<S> for ApiQuery<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        match axum::extract::Query::<T>::from_request_parts(parts, state).await {
            Ok(axum::extract::Query(value)) => Ok(ApiQuery(value)),
            Err(rej) => Err(map_query_rejection(rej)),
        }
    }
}

fn map_query_rejection(rej: QueryRejection) -> ApiError {
    bad_extract("invalid query string", rej)
}

fn bad_extract(prefix: &'static str, rej: impl std::fmt::Display) -> ApiError {
    ApiError::Bad(format!("{prefix}: {rej}"))
}
