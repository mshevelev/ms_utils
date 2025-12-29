import sqlalchemy as sa

from .method_registration import register_method


@register_method(classes=[sa.engine.reflection.Inspector])
def get_tables_and_views_with_schemas(inspector) -> list[dict]:
    schema_names = inspector.get_schema_names()

    res = []
    for schema_name in schema_names:
        table_names = inspector.get_table_names(schema=schema_name)
        for table_name in table_names:
            res.append({"object": "table", "schema": schema_name, "name": table_name})
        for view_name in inspector.get_view_names(schema=schema_name):
            res.append({"object": "view", "schema": schema_name, "name": view_name})
    return res


@register_method(classes=[sa.engine.base.Engine])
def inspect(engine: sa.engine.base.Engine) -> sa.engine.reflection.Inspector:
    """Create an inspector for an `engine`"""
    return sa.inspect(engine)
