from pydantic import BaseModel


class InputModel(BaseModel):
    tree_dbh: int
    curb_loc: object
    user_type: object
    borough: object
    sidewalk: object
    guards: object
    spc_latin: object
    steward: object
    root_stone: object
    root_grate: object
    root_other: object
    trunk_wire: object
    trnk_light: object
    trnk_other: object
    brch_light: object
    brch_shoe: object
    brch_other: object
