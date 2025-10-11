import enum


class CONSTANTS(str, enum.Enum):
    NOT_FOUND = "We don't have an account associated with this email address. Please double-check your email or create a new account."
    NOT_FOUND_2 = "This email doesn’t match our records. Please recheck or sign up to get started."

    NOT_FILLED = "Please complete all required fields to continue."

    PIN_SUCCESS = "Great! Your PIN has been verified successfully. You're good to go."

    INVALID_OR_EXPIRE_PIN = "Oops! That code didn’t work. It may be incorrect or expired—please try again with a new one."
    INVALID_OR_EXPIRE_PIN_2 = "Your code seems incorrect or no longer valid. Please enter the right one or request a new code."

    INVALID_PHONE = "It looks like there’s no account connected to this phone number!"

    INCORRECT_PASSWORD = "The password you entered is incorrect."

    ACCOUNT_FOUND = "Looks like you already have an account. Try logging in!"
    EMAIL_FOUND = "This email number is already linked to an existing account. Try logging in or using a different email!"


class COIN(str, enum.Enum):
    EARN_VERIFY_EMAIL = 30
    EARN_UPDDATE_CONTACT = 30


class RESPONSE_STATUS(str, enum.Enum):
    ACCOUNT_EXIST = "account_registered"
    NOT_FOUND = "no_account_found"
    INSUFFICIENT_BALANCE = "insufficient_balance"
