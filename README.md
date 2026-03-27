# microplex-uk

Thin UK-specific adapters and benchmark harnesses for `microplex`.

Current scope:
- load UK FRS-style PolicyEngine datasets as `ObservationFrame`s
- load UK SPI tax-unit extracts as `ObservationFrame`s
- load UK WAS household wealth extracts as `ObservationFrame`s
- expose UK targets through the canonical `TargetProvider` interface
- benchmark candidate datasets against PolicyEngine UK targets

The intended source analogs are explicit:
- `CPS` ↔ `FRS` (`HOUSEHOLD_INCOME`)
- `PUF` ↔ `SPI` (`TAX_MICRODATA`)
- `SCF` ↔ `WAS` (`WEALTH`)
- `CEX` ↔ `LCFS` (`CONSUMPTION`)

This package is intentionally thin. It is a country-pack spike to test whether
the core `microplex` abstractions generalize cleanly beyond the US.
