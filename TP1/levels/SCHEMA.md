# Grid World Level File Format (SCHEMA)

A Grid World level file is a UTF-8 JSON file residing under `TP1/levels/` that declares a board's dimensions, obstacles, starting car positions, and target flag positions.

## Fields

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `name` | string | Yes | Human-readable name for the level. |
| `rows` | integer | Yes | Number of rows in the grid (1 to 100). |
| `cols` | integer | Yes | Number of columns in the grid (1 to 100). |
| `obstacles` | list of coordinates | Yes | List of black obstacle grid cells. |
| `cars` | list of objects | Yes | List of starting car positions. Each object must have `number` (positive int) and `at` (coordinate). |
| `flags` | list of objects | Yes | List of destination flag positions. Each object must have `number` (positive int) and `at` (coordinate). |

## Coordinate Convention

Coordinates are declared as two-element arrays: `[row, col]`

- **`row 0`** is the top row of the grid.
- **`col 0`** is the left-most column of the grid.
- **`[row, col]`** specifies grid row first, then column. It is **not** `[x, y]`.

## Cars and Flags

Cars and flags are declared as separate lists:
- Each car entry has a 1-based `number` and starting `at` coordinate.
- Each flag entry has a matching 1-based `number` and destination `at` coordinate.
- Car number `n` must be moved to flag number `n` to complete the level.

## Limits

- **Grid Dimensions**: Grid sides must be integers between 1 and 100 inclusive (`MAX_DIMENSION = 100`).
- **Rendering**: The pygame renderer enforces a 24-pixel minimum cell size and clips grids larger than the window area.

## Worked Example

Below is the complete content of `TP1/levels/01-warmup.json`:

```json
{
  "name": "Warmup",
  "rows": 5,
  "cols": 5,
  "obstacles": [
    [2, 1],
    [2, 3]
  ],
  "cars": [
    {
      "number": 1,
      "at": [0, 0]
    },
    {
      "number": 2,
      "at": [0, 4]
    }
  ],
  "flags": [
    {
      "number": 1,
      "at": [4, 4]
    },
    {
      "number": 2,
      "at": [4, 0]
    }
  ]
}
```

## Validation errors

*(Integrity validation catalogue added in Plan 02-02)*

---
*Note: Phase 4 delivery README links to this document for the level specification.*
