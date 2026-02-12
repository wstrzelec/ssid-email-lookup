"""SSID/Email lookup module for SMM API integration.

This module provides FastAPI endpoints to retrieve student IDs or email
addresses from the SMM API. Designed to run in Kubernetes with
environment-based configuration.
"""

import json
import logging
import os
import re
from typing import Dict


import requests
from fastapi import FastAPI, HTTPException  # pylint: disable=import-error
# pylint: disable=no-name-in-module
from pydantic import BaseModel, Field, validator

# Configure logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ssid_email")

# Initialize FastAPI app
app = FastAPI(
    title="SSID/Email Lookup Service",
    description="API for looking up SANS student IDs and email addresses",
    version="2.0.0"
)

# Constants
STUDENT_NOT_FOUND = "Student not found"
REQUEST_TIMEOUT = int(os.environ.get("API_TIMEOUT", "5"))
SMM_BASE_URL = "https://api.sans.org/smm/v1/labs/accounts"
USER_AGENT = "SANS-K8s-SSID-Lookup/2.0"
SMM_API_KEY = os.environ.get("SMM_API_KEY", "")
SMM_API_PASSWORD = os.environ.get("SMM_API_PASSWORD", "")
MAX_INPUT_LENGTH = 255
EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

# Global session for connection pooling
_session = None  # pylint: disable=invalid-name


# Custom exceptions
class StudentNotFoundError(Exception):
    """Raised when a student is not found in the SMM API."""


def _get_session() -> requests.Session:
    """Get or create a requests session for connection pooling.

    Returns:
        Requests session object.
    """
    global _session  # pylint: disable=global-statement
    if _session is None:
        _session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10, pool_maxsize=10, max_retries=3
        )
        _session.mount("http://", adapter)
        _session.mount("https://", adapter)
    return _session


def get_credentials() -> Dict[str, str]:
    """Retrieve credentials from environment variables.

    Returns:
        The credentials as a dictionary.

    Raises:
        RuntimeError: If required environment variables are not set.
    """
    if not SMM_API_KEY or not SMM_API_PASSWORD:
        error_msg = (
            "SMM_API_KEY and SMM_API_PASSWORD environment variables "
            "must be set"
        )
        logger.error("get_credentials: Missing credentials - %s", error_msg)
        raise RuntimeError(error_msg)

    return {
        "Api_key": SMM_API_KEY,
        "Api_Password": SMM_API_PASSWORD
    }


def _validate_credentials(auth_data: Dict[str, str]) -> None:
    """Validate required credential fields.

    Args:
        auth_data: Dictionary containing API credentials.

    Raises:
        KeyError: If required credentials are missing.
    """
    required_keys = ["Api_key", "Api_Password"]
    missing_keys = [key for key in required_keys if key not in auth_data]
    if missing_keys:
        raise KeyError(
            f"Missing required SMM credentials: {', '.join(missing_keys)}"
        )


def _build_request_headers(auth_data: Dict[str, str]) -> Dict[str, str]:
    """Build request headers from authentication data.

    Args:
        auth_data: Dictionary containing API credentials.

    Returns:
        Dictionary containing request headers.
    """
    return {
        "Api-Key": auth_data["Api_key"],
        "Api-Password": auth_data["Api_Password"],
        "User-Agent": USER_AGENT,
    }


def _make_api_request(url: str, headers: Dict[str, str]) -> requests.Response:
    """Make API request and handle basic response validation.

    Args:
        url: API endpoint URL.
        headers: Request headers.

    Returns:
        Response object.

    Raises:
        StudentNotFoundError: If student not found (404).
        RuntimeError: For other API errors.
    """
    # Log request details (exclude sensitive headers)
    safe_headers = {
        k: v for k, v in headers.items()
        if k not in ["Api-Key", "Api-Password"]
    }
    logger.debug(
        "API Request - URL: %s, Headers: %s, Timeout: %ss",
        url, safe_headers, REQUEST_TIMEOUT
    )

    session = _get_session()
    response = session.get(url=url, headers=headers, timeout=REQUEST_TIMEOUT)

    # Log response details
    content_length = len(response.content) if response.content else 0
    logger.debug(
        "API Response - Status: %s, Content-Length: %s bytes",
        response.status_code, content_length
    )

    # Log full response body for debugging
    try:
        logger.debug("API Response Body: %s", response.text)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("Could not log response body: %s", exc)

    # Handle response status
    if response.status_code == 404:
        logger.warning("get_ssid: Student not found")
        raise StudentNotFoundError(STUDENT_NOT_FOUND)

    response.raise_for_status()
    return response


def _parse_api_response(response: requests.Response) -> Dict:
    """Parse and validate API response.

    Args:
        response: HTTP response object.

    Returns:
        Parsed response data.

    Raises:
        RuntimeError: If response parsing or validation fails.
    """
    try:
        response_data = response.json()
        logger.debug(
            "API Response Data (parsed JSON): %s",
            json.dumps(response_data, indent=2)
        )
        data_keys = (
            list(response_data.get("data", {}).keys())
            if isinstance(response_data.get("data"), dict)
            else "No data"
        )
        logger.debug(
            "get_ssid: JSON response parsed - Result: %s, Data keys: %s",
            response_data.get('result', 'Unknown'), data_keys
        )
    except json.JSONDecodeError as exc:
        logger.error(
            "get_ssid: Failed to parse JSON response - Content preview: %s...",
            response.text[:200]
        )
        raise RuntimeError(
            f"Invalid JSON response from SMM API: {str(exc)}"
        ) from exc

    if (not isinstance(response_data, dict) or
            response_data.get("result") != "success"):
        error_result = response_data.get("result", "Unknown error")
        raise RuntimeError(f"SMM API request failed: {error_result}")

    data = response_data.get("data")
    logger.debug("get_ssid: Extracted data from API response - Data: %s", data)
    if not isinstance(data, dict):
        raise RuntimeError(
            f"Invalid data field in API response: {type(data).__name__}"
        )

    return data


def _extract_result_from_data(data: Dict, is_email: bool) -> str:
    """Extract the appropriate result field from API response data.

    Args:
        data: API response data dictionary.
        is_email: True if input was email, False if SSID.

    Returns:
        The extracted result string.

    Raises:
        RuntimeError: If required field is missing.
    """
    if is_email:
        result = data.get("account_id")
        if result is None:
            raise RuntimeError("No account_id found in response")
        return str(result)

    result = data.get("email_address")
    if not result:
        raise RuntimeError("No email_address found in response")
    return result


def _validate_input(email_ssid: str) -> tuple[str, bool]:
    """Validate and process input email or SSID.

    Args:
        email_ssid: The email address or SSID to look up.

    Returns:
        Tuple of (cleaned_input, is_email).

    Raises:
        ValueError: If input is invalid.
    """
    if not email_ssid or not email_ssid.strip():
        raise ValueError("Email/SSID parameter cannot be empty or whitespace")

    email_ssid = email_ssid.strip()

    if len(email_ssid) > MAX_INPUT_LENGTH:
        raise ValueError(
            f"Input exceeds maximum length of {MAX_INPUT_LENGTH} characters"
        )

    is_email = "@" in email_ssid
    if is_email:
        if not EMAIL_PATTERN.match(email_ssid):
            raise ValueError(f"Invalid email format: {email_ssid}")
    else:
        if not email_ssid.isdigit():
            raise ValueError(f"SSID must be numeric, got: {email_ssid}")
        if len(email_ssid) > 10:
            raise ValueError(f"SSID too long, got: {email_ssid}")

    return email_ssid, is_email


def get_ssid(email_ssid: str) -> str:
    """Retrieve the SSID or email address from the SMM API.

    Args:
        email_ssid: The email address or SSID to look up.

    Returns:
        The converted identifier or "Student not found" if 404 error.

    Raises:
        ValueError: If email_ssid is empty or invalid.
        RuntimeError: If API request fails or returns invalid data.
    """
    # Validate and process input
    try:
        email_ssid, is_email = _validate_input(email_ssid)
    except ValueError as exc:
        logger.error(
            "get_ssid: Invalid input - %s - input: %s",
            str(exc), email_ssid
        )
        raise

    input_type = "email" if is_email else "ssid"
    logger.info(
        "get_ssid: Starting lookup - type: %s, length: %s",
        input_type, len(email_ssid)
    )

    try:
        # Get authentication credentials
        auth_data = get_credentials()

        # Validate required credential fields
        _validate_credentials(auth_data)

        # Prepare request headers and URL
        headers = _build_request_headers(auth_data)

        # Construct API endpoint
        if is_email:
            url = f"{SMM_BASE_URL}?email={email_ssid}"
            lookup_type = "email-to-ssid"
        else:
            url = f"{SMM_BASE_URL}/{email_ssid}"
            lookup_type = "ssid-to-email"

        logger.info(
            "get_ssid: Making API request - lookup_type: %s", lookup_type
        )

        # Make API request, parse response, and extract result
        response = _make_api_request(url, headers)
        data = _parse_api_response(response)
        result = _extract_result_from_data(data, is_email)

        logger.info(
            "get_ssid: Lookup successful - lookup_type: %s", lookup_type
        )
        return result

    except requests.exceptions.Timeout as e:
        raise RuntimeError(f"SMM API request timeout: {str(e)}") from e
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"SMM API connection error: {str(e)}") from e
    except requests.exceptions.HTTPError as e:
        if e.response and e.response.status_code == 404:
            raise StudentNotFoundError(STUDENT_NOT_FOUND) from e
        raise RuntimeError(f"SMM API HTTP error: {str(e)}") from e
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"SMM API request error: {str(e)}") from e
    except (KeyError, ValueError, RuntimeError, StudentNotFoundError):
        raise
    except Exception as e:
        raise RuntimeError(f"Unexpected error in get_ssid: {str(e)}") from e


# Pydantic models for request/response
class LookupRequest(BaseModel):  # pylint: disable=too-few-public-methods
    """Request model for SSID/email lookup."""
    student_id: str = Field(
        ...,
        description="Student ID (SSID) or email address to look up"
    )

    @validator('student_id')
    def validate_student_id(cls, v):  # pylint: disable=no-self-argument
        """Validate student_id field."""
        if not v or not v.strip():
            raise ValueError("student_id cannot be empty")

        v = v.strip()
        if len(v) > MAX_INPUT_LENGTH:
            raise ValueError(
                f"Input exceeds maximum length of {MAX_INPUT_LENGTH}"
            )

        # Validate format
        is_email = "@" in v
        if is_email:
            if not EMAIL_PATTERN.match(v):
                raise ValueError(f"Invalid email format: {v}")
        elif not v.isdigit():
            raise ValueError(
                f"Invalid format - must be email or numeric SSID, got: '{v}'"
            )

        return v


class LookupResponse(BaseModel):  # pylint: disable=too-few-public-methods
    """Response model for SSID/email lookup."""
    result: str = Field(
        ...,
        description="Lookup result (converted ID or email)"
    )
    input_type: str = Field(
        ...,
        description="Type of input provided (email or ssid)"
    )


class ErrorResponse(BaseModel):  # pylint: disable=too-few-public-methods
    """Error response model."""
    error: str = Field(..., description="Error message")


@app.get("/health")
async def health_check():
    """Health check endpoint for Kubernetes probes."""
    return {"status": "healthy", "service": "ssid-email-lookup"}


@app.post("/lookup", response_model=LookupResponse)
async def lookup_student(request: LookupRequest):
    """Look up student SSID or email address.

    Args:
        request: Lookup request containing student_id

    Returns:
        LookupResponse with the converted identifier

    Raises:
        HTTPException: For various error conditions
    """
    student_id = request.student_id.strip()
    is_email = "@" in student_id
    input_type = "email" if is_email else "ssid"

    logger.info(
        "Processing lookup request - type: %s, id: %s", input_type, student_id
    )

    try:
        result = get_ssid(student_id)

        # Format response based on input type
        if is_email:
            formatted_result = f"Student ID: {result}"
        else:
            formatted_result = f"Student Email: {result}"

        logger.info("Lookup successful - type: %s", input_type)
        return LookupResponse(result=formatted_result, input_type=input_type)

    except StudentNotFoundError as exc:
        logger.warning(
            "Student not found - type: %s, id: %s", input_type, student_id
        )
        raise HTTPException(status_code=404, detail=STUDENT_NOT_FOUND) from exc
    except ValueError as exc:
        logger.error("Validation error - %s", str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("Runtime error - %s", str(exc))
        raise HTTPException(
            status_code=500,
            detail=f"Lookup failed: {str(exc)}"
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error - %s", str(exc), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(exc)}"
        ) from exc


if __name__ == "__main__":
    import uvicorn  # pylint: disable=import-error

    # Run the FastAPI app with uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        log_level=LOG_LEVEL.lower()
    )
