from strategy_api.db.models import Base, ChatMessage
from strategy_api.db.session import make_engine, make_session_factory


def test_chat_message_roundtrip(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path/'t.db'}")
    Base.metadata.create_all(engine)
    Session = make_session_factory(engine)
    with Session() as s:
        s.add(ChatMessage(session_id="abc", role="user", content="hi", model="m"))
        s.commit()
    with Session() as s:
        rows = s.query(ChatMessage).all()
        assert len(rows) == 1 and rows[0].content == "hi"
        assert isinstance(rows[0].created_at, int)
