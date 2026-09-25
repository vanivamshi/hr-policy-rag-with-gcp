from hr_rag.auth import is_email_allowed


def test_domain_allow_list():
    assert is_email_allowed("Jane@Corp.com", "", "corp.com")
    assert not is_email_allowed("jane@evil.com", "", "corp.com")
    assert not is_email_allowed("jane@notcorp.com", "", "corp.com")


def test_explicit_emails_and_at_prefix():
    assert is_email_allowed("contractor@gmail.com", "contractor@gmail.com", "@corp.com")
    assert is_email_allowed("a@corp.com", "contractor@gmail.com", "@corp.com")


def test_fails_closed():
    assert not is_email_allowed("a@corp.com", "", "")
    assert not is_email_allowed(None, "", "corp.com")
    assert not is_email_allowed("not-an-email", "", "corp.com")
