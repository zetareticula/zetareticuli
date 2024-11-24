use actix_web::{web, HttpResponse, Responder};
use serde::{Deserialize, Serialize};

// Define the request type
#[derive(Deserialize)]
pub struct YourRequestType {
    field1: String,
    field2: i32,
    field3: bool,
}

// Define the response type
#[derive(Serialize)]
pub struct YourResponseType {
    field1: String,
    field2: i32,
    field3: bool,
}

// Define the handler function
async fn handle_request(req: web::Json<YourRequestType>) -> impl Responder {
    // Process the request and generate a response
    let response = YourResponseType {
        field1: req.field1.clone(),
        field2: req.field2,
        field3: req.field3,
    };

    // Return the response as JSON
    HttpResponse::Ok().json(response)
}

