def _parse_csv(value: str) -> set[str]:
    return {item.strip().lower().lstrip("@") for item in value.split(",") if item.strip()}


def is_email_allowed(email: str | None, allowed_emails: str, allowed_domains: str) -> bool:
    """Employee allow-list check. Fails closed: an empty allow-list admits nobody."""
    if not email or "@" not in email:
        return False
    email = email.strip().lower()
    emails = _parse_csv(allowed_emails)
    domains = _parse_csv(allowed_domains)
    return email in emails or email.rsplit("@", 1)[1] in domains
