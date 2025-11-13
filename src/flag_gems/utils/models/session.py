from typing_extensions import override

import sqlalchemy.exc
import sqlalchemy.orm


class RollbackSession(sqlalchemy.orm.Session):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @override
    def commit(self) -> None:
        try:
            super().commit()
        except sqlalchemy.exc.IntegrityError:
            self.rollback()
