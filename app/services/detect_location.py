import requests
import httpx
import socket
from typing import Dict, Optional, Union
from dataclasses import dataclass
import geoip2.database
import geoip2.errors
from datetime import datetime
import pytz

@dataclass
class LocationDetails:
    country: str
    country_code: str
    calling_code: str
    city: str

class GetLocation:
    def __init__(self, request_headers: Dict[str, str], remote_address: str):
        """
        Initialize location detector with request data

        Args:
            request_headers: Dictionary of HTTP headers (e.g., from Flask request.headers)
            remote_address: Remote IP address (e.g., from Flask request.remote_addr)
        """
        # Get IP from X-Forwarded-For header or remote address
        forwarded_for = request_headers.get("x-forwarded-for") or request_headers.get(
            "X-Forwarded-For"
        )
        if forwarded_for:
            # Take the first IP if multiple are present
            self.ip = forwarded_for.split(",")[0].strip()
        else:
            self.ip = remote_address

        self.timezone = request_headers.get("timezone") or request_headers.get(
            "Timezone"
        )

        # Handle localhost/private IP fallbacks
        self._handle_localhost_fallback()

    def _handle_localhost_fallback(self):
        """Handle localhost and private IP addresses by getting public IP"""
        private_ips = ["127.0.0.1", "::1", "localhost"]

        # Check if IP is localhost or private
        if (
            self.ip in private_ips
            or self.ip.startswith("192.168.")
            or self.ip.startswith("10.")
            or self.ip.startswith("172.")
        ):

            try:
                # Get public IP for localhost/development
                public_ip = self.get_public_ip_address()
                if public_ip:
                    self.ip = public_ip
                else:
                    # Fallback to a default IP for testing (Google DNS)
                    self.ip = "8.8.8.8"
            except Exception as e:
                self.ip = "8.8.8.8"  # Fallback

    @property
    def get_timezone(self) -> str:
        """Get timezone from header or system default"""
        if self.timezone:
            return str(self.timezone)

        # Fallback to system timezone
        try:
            return str(datetime.now().astimezone().tzinfo)
        except:
            return "UTC"

    async def get_current_details(self) -> Optional[LocationDetails]:
        """
        Get location details from ipapi.co service (FastAPI async version)

        Returns:
            LocationDetails object with country, city, etc.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://ipapi.co/{self.ip}/json/",
                    timeout=10.0,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    },
                )
                response.raise_for_status()
                data = response.json()

                # Handle API errors
                if "error" in data:
                    return self._get_fallback_location()

                return LocationDetails(
                    country=data.get("country_name", "Unknown"),
                    country_code=data.get("country_code", "XX"),
                    calling_code=data.get("country_calling_code", ""),
                    city=data.get("city", "Unknown"),
                )

        except httpx.RequestError as e:
            return self._get_fallback_location()
        except Exception as e:
            return self._get_fallback_location()

    def _get_fallback_location(self) -> LocationDetails:
        """Provide fallback location data for development/errors"""
        return LocationDetails(
            country="Unknown", country_code="XX", calling_code="", city="Unknown"
        )

    def get_public_ip_address(self) -> Optional[str]:
        """
        Get public IP address using ipify service

        Returns:
            Public IP address as string
        """
        try:
            response = requests.get("https://api.ipify.org", timeout=5)
            response.raise_for_status()
            ip = response.text.strip()
            return ip
        except requests.exceptions.RequestException as e:
            try:
                # Fallback to alternative service
                response = requests.get("https://httpbin.org/ip", timeout=5)
                response.raise_for_status()
                data = response.json()
                ip = data.get("origin", "").split(",")[0].strip()
                return ip
            except:
                return None

def get_current_info(ip: str) -> Optional[str]:
    """
    Get country code from IP using geoip2 (requires MaxMind database)

    Args:
        ip: IP address to lookup

    Returns:
        Country code or None
    """
    try:
        # You'll need to download the GeoLite2-Country.mmdb file from MaxMind
        with geoip2.database.Reader("GeoLite2-Country.mmdb") as reader:
            response = reader.country(ip)
            return response.country.iso_code
    except geoip2.errors.AddressNotFoundError:
        return None
    except FileNotFoundError:
        return None
    except Exception as e:
        return None

def get_current_info_fallback(ip: str) -> Optional[str]:
    """
    Fallback method using online service instead of local database

    Args:
        ip: IP address to lookup

    Returns:
        Country code or None
    """
    try:
        response = requests.get(f"https://ipapi.co/{ip}/country_code/", timeout=5)
        response.raise_for_status()
        country_code = response.text.strip()
        return country_code if country_code != "Undefined" else None
    except Exception as e:
        return None
