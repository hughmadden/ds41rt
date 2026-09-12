#!/usr/bin/env python3
"""Apply audited Unicode FSM fixes to a generated copy of the pinned source."""
import argparse
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--replacement',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
source=a.source.read_text()
old='AddSameLengthCharacterRange(fsm, tmp_state_max, to, 0x0080, (max & 0x00FFFF));'
assert source.count(old)==1,'Pinned UTF-8 range implementation changed'
source=source.replace(old,old.replace('0x0080,','0x008080,'))
begin='FSMWithStartEnd GrammarFSMBuilderImpl::BuildNegativeCharacterClass(const GrammarExpr& expr) {'
end='FSMWithStartEnd GrammarFSMBuilderImpl::CharacterClass(const GrammarExpr& expr) {'
assert source.count(begin)==1 and source.count(end)==1,'Pinned character-class implementation changed'
first,last=source.index(begin),source.index(end)
assert first<last
source=source[:first]+a.replacement.read_text()+'\n'+source[last:]
if not a.output.exists() or a.output.read_text()!=source:a.output.write_text(source)
